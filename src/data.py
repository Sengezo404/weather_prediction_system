from pathlib import Path
import pandas as pd
import numpy as np

from .config import BASE_WEATHER_COLUMNS


def load_nasa_power_csv(path: str | Path, skiprows: int = 15) -> pd.DataFrame:
    """Load the NASA POWER daily CSV used by the project."""
    df = pd.read_csv(path, skiprows=skiprows)

    required = {"YEAR", "DOY"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required NASA POWER columns: {sorted(missing)}")

    df["DATE"] = (
        pd.to_datetime(df["YEAR"].astype(int).astype(str), format="%Y")
        + pd.to_timedelta(df["DOY"].astype(int) - 1, unit="D")
    )

    df = df.set_index("DATE").drop(columns=["YEAR", "DOY"], errors="ignore")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower().str.strip()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.replace(-999.0, np.nan)

    available = [c for c in BASE_WEATHER_COLUMNS if c in df.columns]
    if len(available) < 3:
        raise ValueError(
            "Too few expected weather columns were found. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw weather observations without turning missing data into fake zeros."""
    df = df.copy().sort_index()

    df = df.replace(-999.0, np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.interpolate(method="time", limit_direction="both")

    available = [c for c in BASE_WEATHER_COLUMNS if c in df.columns]
    df = df.dropna(subset=available)

    return df


def prepare_raw_data(path: str | Path, skiprows: int = 15) -> pd.DataFrame:
    return clean_weather_data(load_nasa_power_csv(path, skiprows=skiprows))


def last_valid_date(df: pd.DataFrame, columns: list[str]) -> pd.Timestamp:
    """
    Last date where at least one of `columns` has real (non-NaN) data.

    NASA POWER files always include a trailing block of placeholder rows
    (-999.0, already converted to NaN by this point) for the most recent
    days it hasn't backfilled yet, even when the calendar date isn't in
    the future. Use this to trim that tail off before treating the file's
    max index as the "real" end of the record.
    """
    cols = [c for c in columns if c in df.columns]
    valid_mask = df[cols].notna().any(axis=1)
    if not valid_mask.any():
        raise ValueError("No valid (non-placeholder) rows found in this dataframe.")
    return df.index[valid_mask][-1]


def merge_sources(nasa_df: pd.DataFrame, meteostat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine NASA POWER (primary) and meteostat (gap-filler) into one series.
    Where NASA POWER has a value, it wins; meteostat only fills NaNs.
    """
    combined = nasa_df.combine_first(meteostat_df)
    combined = combined[[c for c in BASE_WEATHER_COLUMNS if c in combined.columns]]
    return clean_weather_data(combined)


_METEOSTAT_COLUMN_MAP = {
    "temp": "t2m",
    "tmax": "t2m_max",
    "tmin": "t2m_min",
    "prcp": "prectotcorr",
    "rhum": "rh2m",
    "wspd": "ws2m",
    "pres": "ps",
}


def _sea_level_to_station_pressure(pres_hpa: pd.Series, elevation_m: float) -> pd.Series:
    """
    Convert meteostat's sea-level-adjusted pressure to approximate station-level
    (surface) pressure, using the ICAO standard-atmosphere reduction formula.

    NASA POWER's `ps` is surface pressure at the site's actual elevation. Feeding
    a model trained on that scale with meteostat's uncorrected sea-level pressure
    (which reads ~140 hPa too high for a ~1281m-elevation station like Mahikeng)
    is wildly out-of-distribution and can produce extreme, nonsensical predictions
    from any linear component in the model (e.g. Ridge in a voting ensemble).
    """
    if elevation_m is None or pd.isna(elevation_m) or elevation_m <= 0:
        return pres_hpa
    return pres_hpa * (1 - 0.0065 * elevation_m / 288.15) ** 5.255


def load_meteostat_daily(
    lat: float,
    lon: float,
    start,
    end,
    station_search_radius: int = 150_000,
    station_search_limit: int = 5,
    fill: bool = True,
) -> pd.DataFrame:
    from meteostat import Point, daily, stations

    point = Point(lat, lon)
    candidates = stations.nearby(point, radius=station_search_radius, limit=station_search_limit)
    if candidates.empty:
        raise ValueError(f"No meteostat stations found near ({lat}, {lon}).")

    df = pd.DataFrame()
    elevation = None
    for station_id in candidates.index:
        ts = daily(station_id, start=start, end=end)
        candidate_df = ts.fetch(fill=fill)
        if candidate_df is None or candidate_df.empty:
            continue
        df = candidate_df
        elevation = candidates.loc[station_id, "elevation"]
        if df["temp"].isna().mean() < 0.2:
            break

    if df.empty:
        raise ValueError("No usable meteostat data found for any nearby station.")

    df = df.rename(columns=_METEOSTAT_COLUMN_MAP)
    if "ws2m" in df.columns:
        df["ws2m"] = df["ws2m"] / 3.6
    if "ps" in df.columns:
        df["ps"] = _sea_level_to_station_pressure(df["ps"], elevation)

    keep = [c for c in BASE_WEATHER_COLUMNS if c in df.columns]
    df = df[keep]
    df.index.name = "DATE"
    return df


def prepare_meteostat_data(lat: float, lon: float, start, end, **kwargs) -> pd.DataFrame:
    return clean_weather_data(load_meteostat_daily(lat, lon, start, end, **kwargs))


def get_source_data(
    source: str,
    data_path: str | Path,
    lat: float,
    lon: float,
    skiprows: int = 15,
) -> pd.DataFrame:
    """
    Build one cleaned dataset for training, keyed by source name:
    "nasa", "meteostat", or "combined".

    The NASA POWER file's trailing placeholder rows are trimmed off first
    (see last_valid_date) so "meteostat" and "combined" only ever fetch/use
    a date range NASA POWER actually has real observations for.
    """
    nasa_raw = load_nasa_power_csv(data_path, skiprows=skiprows)
    end_date = last_valid_date(nasa_raw, BASE_WEATHER_COLUMNS)
    nasa_raw = nasa_raw.loc[:end_date]

    if source == "nasa":
        return clean_weather_data(nasa_raw)

    meteo_raw = load_meteostat_daily(lat=lat, lon=lon, start=nasa_raw.index.min(), end=end_date)

    if source == "meteostat":
        return clean_weather_data(meteo_raw)
    if source == "combined":
        return merge_sources(nasa_raw, meteo_raw)

    raise ValueError(f"Unknown source '{source}'. Expected 'nasa', 'meteostat', or 'combined'.")