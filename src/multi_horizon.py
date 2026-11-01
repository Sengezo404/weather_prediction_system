"""Direct multi-horizon weather forecasting.

The model is trained for multiple future horizons so that the dashboard
can always produce:
    1. Today's forecast
    2. The following 10 days

The dashboard only displays a 10-day future forecast. Additional horizons
are trained internally so that delayed live observations can still be mapped
to the correct calendar date.

Per-horizon results saved for the dashboard and the thesis:
    * MAE, RMSE and bias for temperature
    * 10th and 90th percentile of the test residuals (actual - predicted),
      used to draw the uncertainty band
    * a persistence baseline ("the weather stays as it is now") and a skill
      score against it
    * for rain: Brier score, a climatology baseline and a Brier skill score
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "fused_weather.csv"

MODEL_DIR = ROOT / "models" / "multi_horizon"

RESULTS_PATH = ROOT / "data" / "processed" / "multi_horizon_results.json"


# ============================================================
# Forecast configuration
# ============================================================

# We train more horizons internally so that the system can handle
# situations where NASA POWER / Meteostat are several days behind.
# The dashboard itself only DISPLAYS today's prediction and the next 10 days.

MAX_HORIZON = 20

HORIZONS = range(1, MAX_HORIZON + 1)

LAGS = (1, 2, 3, 7, 14)

ROLLING = (3, 7)

RAIN_THRESHOLD_MM = 0.1

# Metrics always come from models trained on the first 80% of the data
# (chronological split). When True, the models that are SAVED and used for
# live forecasts are then refitted on all available history, so the newest
# months are not wasted. Reported metrics therefore describe the 80% models.
REFIT_ON_ALL_DATA = True

REG_PARAMS = dict(
    max_iter=250,
    learning_rate=0.06,
    max_leaf_nodes=31,
    l2_regularization=0.2,
    random_state=42,
)


# ============================================================
# Load data
# ============================================================

def load_data(path=DATA_PATH):
    """Load and clean the fused historical dataset."""

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ============================================================
# Select target columns
# ============================================================

def choose_columns(df):

    names = (
        "fused_t2m_max",
        "fused_t2m_min",
        "fused_prectotcorr",
    )

    missing = [c for c in names if c not in df.columns]

    if missing:
        raise ValueError(f"Missing fused target columns: {missing}")

    return names


# ============================================================
# Feature generation
# ============================================================

def make_features(df):

    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    cols = [
        c
        for c in numeric.columns
        if not any(word in c.lower() for word in ("target", "future", "tomorrow"))
    ]

    base = numeric[cols]

    pieces = [base]

    for lag in LAGS:
        pieces.append(base.shift(lag).add_suffix(f"_lag{lag}"))

    for window in ROLLING:
        pieces.append(
            base.rolling(window, min_periods=window).mean().add_suffix(f"_roll{window}")
        )

    return pd.concat(pieces, axis=1).copy()


# ============================================================
# Helpers
# ============================================================

def _temperature_metrics(actual, pred, persistence):
    """Metrics for one temperature model on the test period."""

    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = actual - pred

    mae = float(mean_absolute_error(actual, pred))

    persistence = np.asarray(persistence, dtype=float)
    ok = ~np.isnan(persistence)
    baseline_mae = None
    skill = None
    if ok.sum() >= 10:
        baseline_mae = float(mean_absolute_error(actual[ok], persistence[ok]))
        if baseline_mae > 0:
            skill = float(1 - mae / baseline_mae)

    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "bias": float(residual.mean()),
        "q10": float(np.quantile(residual, 0.10)),
        "q90": float(np.quantile(residual, 0.90)),
        "baseline_mae": baseline_mae,
        "skill": skill,
        "n_test": int(len(actual)),
    }


# ============================================================
# Train multi-horizon models
# ============================================================

def train_multi_horizon(path=DATA_PATH):

    df = load_data(path)

    mx, mn, rain = choose_columns(df)

    X = make_features(df)

    split = int(len(df) * 0.8)

    if split < 100 or len(df) - split < 30:
        raise ValueError("Not enough historical rows.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_columns": list(X.columns),
        "targets": {"max": mx, "min": mn, "rain": rain},
        "max_horizon": MAX_HORIZON,
        "lags": list(LAGS),
        "rolling_windows": list(ROLLING),
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "refit_on_all_data": REFIT_ON_ALL_DATA,
    }

    results = {
        "config": {
            "test_start": str(df.index[split].date()),
            "horizons": list(HORIZONS),
            "max_horizon": MAX_HORIZON,
            "rain_threshold_mm": RAIN_THRESHOLD_MM,
            "refit_on_all_data": REFIT_ON_ALL_DATA,
            "residual_definition": "actual minus predicted",
        },
        "regression": {},
        "classification": {},
    }

    index_numbers = np.arange(len(df))

    for h in HORIZONS:

        y_max = df[mx].shift(-h)
        y_min = df[mn].shift(-h)
        rain_amount = df[rain].shift(-h)

        valid = y_max.notna() & y_min.notna() & rain_amount.notna()

        # Training targets must occur before the chronological boundary.
        train = valid & pd.Series(index_numbers + h < split, index=df.index)
        test = valid & pd.Series(index_numbers >= split, index=df.index)

        if train.sum() < 100 or test.sum() < 10:
            print(f"Skipping horizon {h}: insufficient train/test samples.", flush=True)
            continue

        xtr = X.loc[train]
        xte = X.loc[test]
        xall = X.loc[valid]

        # ====================================================
        # Temperature models
        # ====================================================

        final_models = {}
        reg_results = {}

        for name, y, col in (("max", y_max, mx), ("min", y_min, mn)):

            model = HistGradientBoostingRegressor(**REG_PARAMS)
            model.fit(xtr, y.loc[train])
            pred = model.predict(xte)

            reg_results[f"day_{h}_{name}"] = _temperature_metrics(
                y.loc[test],
                pred,
                df[col].loc[test],  # persistence: today's value repeated
            )

            if REFIT_ON_ALL_DATA:
                final = clone(model).fit(xall, y.loc[valid])
            else:
                final = model
            final_models[name] = final

        # ====================================================
        # Rainfall classification
        # ====================================================

        yr = (rain_amount > RAIN_THRESHOLD_MM).astype(int)

        if yr.loc[train].nunique() != 2:
            print(
                f"Skipping rain model for horizon {h}: "
                "only one rainfall class in training data.",
                flush=True,
            )
            continue

        clf = HistGradientBoostingClassifier(**REG_PARAMS)
        clf.fit(xtr, yr.loc[train])

        prob = clf.predict_proba(xte)[:, 1]
        actual = yr.loc[test]
        predicted = (prob >= 0.5).astype(int)

        brier = float(brier_score_loss(actual, prob))
        base_rate = float(yr.loc[train].mean())
        baseline_brier = float(np.mean((actual.to_numpy() - base_rate) ** 2))

        class_result = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(actual, prob)) if actual.nunique() == 2 else None
            ),
            "brier": brier,
            "baseline_brier": baseline_brier,
            "brier_skill": (
                float(1 - brier / baseline_brier) if baseline_brier > 0 else None
            ),
            "n_test": int(len(actual)),
        }

        final_clf = clone(clf).fit(xall, yr.loc[valid]) if REFIT_ON_ALL_DATA else clf

        # ====================================================
        # Record results and save models (only when all three exist)
        # ====================================================

        results["regression"].update(reg_results)
        results["classification"][f"day_{h}_rain"] = class_result

        joblib.dump(final_models["max"], MODEL_DIR / f"horizon_{h:02d}_max.joblib")
        joblib.dump(final_models["min"], MODEL_DIR / f"horizon_{h:02d}_min.joblib")
        joblib.dump(final_clf, MODEL_DIR / f"horizon_{h:02d}_rain.joblib")

        print(
            f"Trained horizon {h}/{MAX_HORIZON} | "
            f"max MAE={reg_results[f'day_{h}_max']['mae']:.2f} | "
            f"rain F1={class_result['f1']:.3f}",
            flush=True,
        )

    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\nMulti-horizon training completed.", flush=True)

    return results


# ============================================================
# Predict multiple horizons
# ============================================================

def predict_10_days(history):

    metadata_path = MODEL_DIR / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Multi-horizon models have not been trained yet.\n"
            "Run:\n"
            "python -m src.multi_horizon"
        )

    metadata = json.loads(metadata_path.read_text())

    history = history.sort_index().copy()
    history.index = pd.to_datetime(history.index).normalize()

    X = make_features(history)

    missing = set(metadata["feature_columns"]) - set(X.columns)

    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)[:10]}")

    latest = X.loc[:, metadata["feature_columns"]].iloc[[-1]]

    last_observation_date = pd.Timestamp(latest.index[-1]).normalize()

    rows = []

    max_horizon = int(metadata.get("max_horizon", MAX_HORIZON))

    for h in range(1, max_horizon + 1):

        max_path = MODEL_DIR / f"horizon_{h:02d}_max.joblib"
        min_path = MODEL_DIR / f"horizon_{h:02d}_min.joblib"
        rain_path = MODEL_DIR / f"horizon_{h:02d}_rain.joblib"

        # If a particular horizon could not be trained, skip it.
        if not (max_path.exists() and min_path.exists() and rain_path.exists()):
            continue

        tmax = float(joblib.load(max_path).predict(latest)[0])
        tmin = float(joblib.load(min_path).predict(latest)[0])
        prob = float(joblib.load(rain_path).predict_proba(latest)[0, 1])

        if not np.isfinite([tmax, tmin, prob]).all():
            raise ValueError(f"Invalid forecast at horizon {h}")

        if not 0 <= prob <= 1:
            raise ValueError(f"Invalid rain probability at horizon {h}: {prob}")

        rows.append(
            {
                "Horizon": h,
                "Date": last_observation_date + pd.Timedelta(days=h),
                "Maximum Temperature": tmax,
                "Minimum Temperature": tmin,
                "Rain Probability": prob,
                "Rain Expected": prob >= 0.5,
            }
        )

    if not rows:
        raise ValueError("No multi-horizon forecasts could be generated.")

    return pd.DataFrame(rows)


# ============================================================
# Command-line training
# ============================================================

if __name__ == "__main__":

    train_multi_horizon()