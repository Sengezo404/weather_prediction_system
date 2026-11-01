from __future__ import annotations
import json
from datetime import date, timedelta
import joblib
import numpy as np
import pandas as pd
import requests
from meteostat import daily
from .fused_system import (FUSED_MODEL_PATH, FUSED_RESULTS_PATH, FEATURE_SOURCE_COLUMNS, STATION_ID, build_fused_dataset, load_nasa, make_training_table)

LAT, LON = -25.864, 25.644
LOCAL_TZ = "Africa/Johannesburg"
NASA_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,PS"
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"


def _local_today() -> date:
    """Today's date in Mahikeng (the server may run in UTC)."""
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).date()


def fetch_nasa_live(start: date, end: date) -> pd.DataFrame:
    params = {"parameters": NASA_PARAMS, "community": "AG", "longitude": LON, "latitude": LAT, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON", "time-standard": "LST"}
    r = requests.get(NASA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    dates = sorted(set().union(*(v.keys() for v in data.values())))
    df = pd.DataFrame(index=pd.to_datetime(dates, format="%Y%m%d"))
    for k, values in data.items():
        df[k.lower()] = pd.Series(values, dtype="float64").rename(index=lambda x: pd.to_datetime(x, format="%Y%m%d")).reindex(df.index)
    df = df.replace(-999.0, np.nan)
    if "ps" in df:
        df["ps"] *= 10.0
    return df.reindex(columns=FEATURE_SOURCE_COLUMNS).sort_index()


def fetch_meteostat_live(start: date, end: date) -> pd.DataFrame:
    df = daily(STATION_ID, start=start, end=end).fetch(fill=False)
    if df is None or df.empty:
        raise RuntimeError("Meteostat returned no recent observations")
    df = df.rename(columns={"temp": "t2m", "tavg": "t2m", "tmax": "t2m_max", "tmin": "t2m_min", "prcp": "prectotcorr", "rhum": "rh2m", "wspd": "ws2m", "pres": "ps"})
    df.index = pd.to_datetime(df.index).normalize()
    if "ws2m" in df:
        df["ws2m"] /= 3.6
    return df.reindex(columns=FEATURE_SOURCE_COLUMNS).sort_index()


def fetch_openmeteo_recent(start: date, end: date) -> pd.DataFrame:
    """Recent days up to yesterday from Open-Meteo (free, no API key).

    NASA POWER and Meteostat both publish with a delay of several days. Open-Meteo
    is used only to fill the days they have not published yet. Its values are
    model-based estimates, so those days are reported as provisional.
    """
    past_days = int(min(92, max(2, (date.today() - start).days + 2)))
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "hourly": "relative_humidity_2m,wind_speed_10m,surface_pressure",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": LOCAL_TZ,
        "wind_speed_unit": "ms",
    }
    r = requests.get(OPEN_METEO_API, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    d = pd.DataFrame(js["daily"])
    d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("time"))).normalize()
    d = d.apply(pd.to_numeric, errors="coerce")

    h = pd.DataFrame(js["hourly"])
    h.index = pd.DatetimeIndex(pd.to_datetime(h.pop("time")))
    h = h.apply(pd.to_numeric, errors="coerce")
    hourly_mean = h.groupby(h.index.normalize()).mean()

    out = pd.DataFrame(index=d.index)
    out["t2m"] = d["temperature_2m_mean"]
    out["t2m_max"] = d["temperature_2m_max"]
    out["t2m_min"] = d["temperature_2m_min"]
    out["prectotcorr"] = d["precipitation_sum"]
    out["rh2m"] = hourly_mean["relative_humidity_2m"].reindex(out.index)
    out["ws2m"] = hourly_mean["wind_speed_10m"].reindex(out.index)
    out["ps"] = hourly_mean["surface_pressure"].reindex(out.index)
    out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    return out.reindex(columns=FEATURE_SOURCE_COLUMNS).sort_index()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), columns=FEATURE_SOURCE_COLUMNS, dtype="float64")


def _try(fn, *args):
    try:
        return fn(*args), None
    except Exception as exc:  # a source being down must not stop the app
        return None, str(exc)


def _through(df: pd.DataFrame, columns: list[str]) -> str:
    """Last date on which every one of `columns` has a value."""
    present = [c for c in columns if c in df.columns]
    if not present or df.empty:
        return "unavailable"
    ok = df[present].notna().all(axis=1)
    return str(ok[ok].index.max().date()) if ok.any() else "unavailable"


def _has_temps(df: pd.DataFrame) -> pd.Series:
    return df[["t2m_max", "t2m_min"]].notna().all(axis=1)


def prepare_live_fused_history(days: int = 60) -> tuple[pd.DataFrame, dict]:
    end = _local_today() - timedelta(days=1)  # yesterday, Mahikeng time
    start = end - timedelta(days=days - 1)

    nasa = load_nasa()
    nasa_live, nasa_err = _try(fetch_nasa_live, start, end)
    met_live, met_err = _try(fetch_meteostat_live, start, end)
    recent, om_err = _try(fetch_openmeteo_recent, start, end)
    if nasa_live is None and met_live is None and recent is None:
        raise RuntimeError(f"All live sources failed: NASA ({nasa_err}); Meteostat ({met_err}); Open-Meteo ({om_err})")
    nasa_live = _empty() if nasa_live is None else nasa_live
    met_live = _empty() if met_live is None else met_live

    nasa_all = pd.concat([nasa.loc[nasa.index < pd.Timestamp(start)], nasa_live])
    nasa_all = nasa_all[~nasa_all.index.duplicated(keep="last")].sort_index()
    if recent is not None:
        # Fills only the cells NASA POWER has not published yet; existing values are kept.
        nasa_all = nasa_all.combine_first(recent)

    fused = build_fused_dataset(nasa_all, met_live)

    # A day counts as observed once its temperatures are known. Rainfall left
    # missing is imputed by the model pipelines instead of discarding the day.
    has_temps = fused[["fused_t2m_max", "fused_t2m_min"]].notna().all(axis=1)
    fused = fused.loc[has_temps & (fused.index <= pd.Timestamp(end))]
    if fused.empty:
        raise RuntimeError("No recent fused observation with temperatures is available")

    # Days that neither NASA POWER nor Meteostat has published (filled by Open-Meteo).
    published = (_has_temps(nasa_live).reindex(fused.index, fill_value=False)
                 | _has_temps(met_live).reindex(fused.index, fill_value=False))
    provisional = (fused.index >= pd.Timestamp(start)) & ~published.to_numpy()

    rain_known = fused["fused_prectotcorr"].notna()
    rain_through = fused.index[rain_known].max() if rain_known.any() else None
    meta = {
        "nasa_through": _through(nasa_live, ["t2m_max", "t2m_min"]),
        "meteostat_through": _through(met_live, ["t2m_max", "t2m_min"]),
        "rain_through": str(rain_through.date()) if rain_through is not None else "unavailable",
        "rain_missing_days": int((fused.index > rain_through).sum()) if rain_through is not None else int(len(fused)),
        "provisional_days": int(provisional.sum()),
        "provisional_from": str(fused.index[provisional].min().date()) if provisional.any() else None,
        "source_errors": {k: v for k, v in {"NASA POWER": nasa_err, "Meteostat": met_err, "Open-Meteo": om_err}.items() if v},
    }
    return fused, meta


def validate_forecast(tmax: float, tmin: float, rain_probability: float) -> None:
    if not (np.isfinite(tmax) and np.isfinite(tmin) and np.isfinite(rain_probability)):
        raise ValueError("Non-finite prediction")
    if not (-10 <= tmin <= 45 and -5 <= tmax <= 55 and tmin <= tmax + 2):
        raise ValueError(f"Invalid temperature forecast: max={tmax}, min={tmin}")
    if not 0 <= rain_probability <= 1:
        raise ValueError("Invalid rain probability")


def predict_next_day(fused_history: pd.DataFrame) -> dict:
    if not FUSED_MODEL_PATH.exists():
        raise FileNotFoundError("Train the fused model with python -m src.fused_system")
    artifact = joblib.load(FUSED_MODEL_PATH)
    X, _, _, _ = make_training_table(fused_history, require_targets=False)
    latest = X.iloc[[-1]].reindex(columns=artifact["feature_columns"])
    tmax = float(artifact["tmax"].predict(latest)[0])
    tmin = float(artifact["tmin"].predict(latest)[0])
    prob = float(artifact["rain"].predict_proba(latest)[0, 1])
    validate_forecast(tmax, tmin, prob)
    return {"forecast_date": pd.Timestamp(latest.index[-1]).date() + timedelta(days=1), "predicted_max_temperature": tmax, "predicted_min_temperature": tmin, "rain_probability": prob, "rain_tomorrow": prob >= 0.5}


def load_results():
    return json.loads(FUSED_RESULTS_PATH.read_text(encoding="utf-8")) if FUSED_RESULTS_PATH.exists() else None