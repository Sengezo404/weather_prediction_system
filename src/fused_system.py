from __future__ import annotations

import json

from datetime import date, timedelta

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

from meteostat import daily

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor

from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

try:

    from .config import MODEL_DIR, PROCESSED_DATA_PATH

except Exception:

    ROOT = Path(__file__).resolve().parents[1]

    MODEL_DIR = ROOT / "models"

    PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "weather_processed.csv"

ROOT = Path(__file__).resolve().parents[1]

NASA_PATH = ROOT / "data" / "raw" / "mahikeng_weather.csv"

FUSED_PATH = ROOT / "data" / "processed" / "fused_weather.csv"

FUSED_MODEL_PATH = MODEL_DIR / "fused_models.joblib"

FUSED_RESULTS_PATH = MODEL_DIR / "fused_training_results.json"

STATION_ID = "68242"  # Mafikeng Intl. Ad., closest station used for Mahikeng

FEATURE_SOURCE_COLUMNS = ["t2m", "t2m_max", "t2m_min", "prectotcorr", "rh2m", "ws2m", "ps"]

def load_nasa(path: Path = NASA_PATH) -> pd.DataFrame:

    df = pd.read_csv(path, skiprows=15)

    df["DATE"] = pd.to_datetime(df["YEAR"].astype(int).astype(str), format="%Y") + pd.to_timedelta(df["DOY"].astype(int) - 1, unit="D")

    df = df.set_index("DATE")

    df = df.drop(columns=["YEAR", "DOY"], errors="ignore")

    df.columns = df.columns.str.lower().str.strip()

    df = df.replace(-999.0, np.nan)

    for c in FEATURE_SOURCE_COLUMNS:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce")

    # NASA POWER's project CSV stores PS around 86.7 for Mahikeng; normalize

    # pressure to hPa so it is comparable with Meteostat's pressure values.

    if "ps" in df.columns:

        df["ps"] = df["ps"] * 10.0

    return df[[c for c in FEATURE_SOURCE_COLUMNS if c in df.columns]].sort_index()

def load_meteostat(start: date, end: date) -> pd.DataFrame:

    ts = daily(STATION_ID, start=start, end=end)

    df = ts.fetch(fill=False)

    if df is None or df.empty:

        raise RuntimeError("Meteostat returned no daily observations.")

    rename = {

        "temp": "t2m", "tavg": "t2m", "tmax": "t2m_max", "tmin": "t2m_min",

        "prcp": "prectotcorr", "rhum": "rh2m", "wspd": "ws2m", "pres": "ps",

    }

    df = df.rename(columns=rename)

    if "ws2m" in df.columns:

        df["ws2m"] = df["ws2m"] / 3.6  # km/h -> m/s

    for c in FEATURE_SOURCE_COLUMNS:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Meteostat pressure is already in hPa.

    df.index = pd.to_datetime(df.index).normalize()

    return df[[c for c in FEATURE_SOURCE_COLUMNS if c in df.columns]].sort_index()

def build_fused_dataset(nasa: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:

    idx = nasa.index.union(met.index).sort_values()

    out = pd.DataFrame(index=idx)

    for c in FEATURE_SOURCE_COLUMNS:

        out[f"nasa_{c}"] = nasa[c].reindex(idx) if c in nasa else np.nan

        out[f"met_{c}"] = met[c].reindex(idx) if c in met else np.nan

        a, b = out[f"nasa_{c}"], out[f"met_{c}"]

        out[f"fused_{c}"] = pd.concat([a, b], axis=1).median(axis=1, skipna=True)

        out[f"diff_{c}"] = (a - b).abs()

    # Only interpolate short gaps. Do not manufacture long historical periods.

    for c in out.columns:

        if c.startswith(("nasa_", "met_", "fused_", "diff_")):

            out[c] = out[c].interpolate(method="time", limit=3, limit_area="inside")

    return out

def make_training_table(fused: pd.DataFrame, require_targets: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:

    base = fused.copy()

    for c in FEATURE_SOURCE_COLUMNS:

        fc = f"fused_{c}"

        base[f"{fc}_lag1"] = base[fc].shift(1)

        base[f"{fc}_lag2"] = base[fc].shift(2)

        base[f"{fc}_lag3"] = base[fc].shift(3)

        base[f"{fc}_lag7"] = base[fc].shift(7)

        base[f"{fc}_roll3"] = base[fc].shift(1).rolling(3, min_periods=2).mean()

        base[f"{fc}_roll7"] = base[fc].shift(1).rolling(7, min_periods=3).mean()

        base[f"{fc}_roll14"] = base[fc].shift(1).rolling(14, min_periods=7).mean()

    doy = base.index.dayofyear

    base["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)

    base["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    base["month"] = base.index.month

    base["day_of_year"] = doy

    # Targets are next-day consensus values from both sources where available.

    target_max = base["fused_t2m_max"].shift(-1)

    target_min = base["fused_t2m_min"].shift(-1)

    target_rain_amount = base["fused_prectotcorr"].shift(-1)

    target_rain = (target_rain_amount > 0.1).astype(float)

    # Features: source-specific observations + fused values + source agreement.

    feature_cols = [c for c in base.columns if not c.startswith("fused_t2m_max") and not c.startswith("fused_t2m_min")]

    # Exclude all same-day target variables from feature set only when they are future-derived; current day's values are legitimate.

    feature_cols = [c for c in feature_cols if c not in {"fused_t2m_max", "fused_t2m_min"}]

    # Keep current-day max/min as predictors under source-specific names and fused rainfall etc.

    feature_cols += ["fused_t2m_max", "fused_t2m_min"]

    feature_cols = list(dict.fromkeys(feature_cols))

    X = base[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Training requires known next-day targets. Inference must retain the
    # latest observation, whose next-day targets are not yet available.
    if require_targets:
        valid = target_max.notna() & target_min.notna() & target_rain_amount.notna()
        X = X.loc[valid]
        return X, target_max.loc[valid], target_min.loc[valid], target_rain.loc[valid]

    return X, target_max, target_min, target_rain

def make_regressors():

    ridge = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])

    extra = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, random_state=42, n_jobs=-1))])

    rf = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42, n_jobs=-1))])

    hgb = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42))])

    return {"ridge": ridge, "extra_trees": extra, "random_forest": rf, "hist_gradient_boosting": hgb}

def train_and_save():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    FUSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    nasa = load_nasa()

    start = max(nasa.index.min().date(), date(2005, 1, 1))

    end = date.today() - timedelta(days=1)

    print(f"NASA POWER: {nasa.index.min().date()} -> {nasa.index.max().date()} ({len(nasa):,} rows)")

    print(f"Downloading Meteostat: {start} -> {end} from station {STATION_ID} ...")

    met = load_meteostat(start, end)

    print(f"Meteostat: {met.index.min().date()} -> {met.index.max().date()} ({len(met):,} rows)")

    fused = build_fused_dataset(nasa, met)

    fused.to_csv(FUSED_PATH)

    X, y_max, y_min, y_rain = make_training_table(fused)

    # Strict chronological holdout: last 20% is never used for training.

    split = int(len(X) * 0.80)

    X_train, X_test = X.iloc[:split - 1], X.iloc[split:]

    ym_tr, ym_te = y_max.iloc[:split - 1], y_max.iloc[split:]

    yn_tr, yn_te = y_min.iloc[:split - 1], y_min.iloc[split:]

    yr_tr, yr_te = y_rain.iloc[:split - 1], y_rain.iloc[split:]

    models_max = make_regressors()

    models_min = make_regressors()

    fitted_max, fitted_min = {}, {}

    results = {"regression": {}, "classification": {}, "data": {}}

    for name, model in models_max.items():

        model.fit(X_train, ym_tr)

        pred = model.predict(X_test)

        fitted_max[name] = model

        results["regression"][f"tmax_{name}"] = {

            "mae": float(mean_absolute_error(ym_te, pred)),

            "rmse": float(np.sqrt(mean_squared_error(ym_te, pred))),

            "r2": float(r2_score(ym_te, pred)),

        }

    for name, model in models_min.items():

        model.fit(X_train, yn_tr)

        pred = model.predict(X_test)

        fitted_min[name] = model

        results["regression"][f"tmin_{name}"] = {

            "mae": float(mean_absolute_error(yn_te, pred)),

            "rmse": float(np.sqrt(mean_squared_error(yn_te, pred))),

            "r2": float(r2_score(yn_te, pred)),

        }

    # Voting ensemble of the four independently trained regressors.

    ens_max = VotingRegressor([(k, v) for k, v in make_regressors().items()])

    ens_min = VotingRegressor([(k, v) for k, v in make_regressors().items()])

    ens_max.fit(X_train, ym_tr)

    ens_min.fit(X_train, yn_tr)

    pmax = ens_max.predict(X_test)

    pmin = ens_min.predict(X_test)

    results["regression"]["tmax_ensemble"] = {"mae": float(mean_absolute_error(ym_te, pmax)), "rmse": float(np.sqrt(mean_squared_error(ym_te, pmax))), "r2": float(r2_score(ym_te, pmax))}

    results["regression"]["tmin_ensemble"] = {"mae": float(mean_absolute_error(yn_te, pmin)), "rmse": float(np.sqrt(mean_squared_error(yn_te, pmin))), "r2": float(r2_score(yn_te, pmin))}

    rain_models = {

        "logistic": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),

        "hist_gradient_boosting": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42))]),

    }

    fitted_rain = {}

    for name, model in rain_models.items():

        model.fit(X_train, yr_tr.astype(int))

        prob = model.predict_proba(X_test)[:, 1]

        pred = (prob >= 0.5).astype(int)

        fitted_rain[name] = model

        results["classification"][name] = {

            "accuracy": float(accuracy_score(yr_te, pred)),

            "precision": float(precision_score(yr_te, pred, zero_division=0)),

            "recall": float(recall_score(yr_te, pred, zero_division=0)),

            "f1": float(f1_score(yr_te, pred, zero_division=0)),

            "roc_auc": float(roc_auc_score(yr_te, prob)) if yr_te.nunique() > 1 else None,

        }

    results["data"] = {

        "nasa_start": str(nasa.index.min().date()),

        "nasa_end": str(nasa.index.max().date()),

        "meteostat_start": str(met.index.min().date()),

        "meteostat_end": str(met.index.max().date()),

        "fused_rows": int(len(fused)),

        "training_rows": int(len(X_train)),

        "test_rows": int(len(X_test)),

        "station": STATION_ID,

        "target_definition": "Next-day median/consensus of available NASA POWER and Meteostat values.",

    }

    artifact = {

        "tmax": ens_max,

        "tmin": ens_min,

        "rain": fitted_rain["hist_gradient_boosting"],

        "feature_columns": list(X.columns),

        "train_end": str(X_train.index.max().date()),

        "test_start": str(X_test.index.min().date()),

        "data_end": str(fused.index.max().date()),

    }

    joblib.dump(artifact, FUSED_MODEL_PATH)

    FUSED_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== FUSED MODEL RESULTS ===")

    for k, v in results["regression"].items():

        print(f"{k:28s} MAE={v['mae']:.3f} RMSE={v['rmse']:.3f} R2={v['r2']:.3f}")

    for k, v in results["classification"].items():

        print(f"rain_{k:22s} ACC={v['accuracy']:.3f} F1={v['f1']:.3f} AUC={v['roc_auc']}")

    print(f"\nSaved: {FUSED_MODEL_PATH}")

    print(f"Saved: {FUSED_PATH}")

if __name__ == "__main__":

    train_and_save()
