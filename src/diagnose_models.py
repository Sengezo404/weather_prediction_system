"""
diagnose_models.py

Runs EACH individual saved model (not just the "ensemble") on the same
feature row app.py would use for tomorrow's forecast, so we can see
exactly which model is producing a runaway prediction, plus surfaces
each model's backtested MAE/F1 from training_results.json for comparison.

Usage:
    python diagnose_models.py
"""
import json

import joblib
import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DATA_PATH, STATION_LAT, STATION_LON, BASE_WEATHER_COLUMNS
from src.data import clean_weather_data, prepare_meteostat_data
from src.features import build_features


def build_live_forecast_input(full_history: pd.DataFrame, live_tail: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in BASE_WEATHER_COLUMNS if c in full_history.columns]
    base_history = full_history[base_cols]
    combined = live_tail.combine_first(base_history).sort_index()
    full_range = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
    combined = combined.reindex(full_range)
    combined.index.name = "DATE"
    return clean_weather_data(combined)


def main():
    history = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    print(f"Static history: {history.index.min().date()} to {history.index.max().date()} ({len(history)} rows)")
    today = pd.Timestamp.today().normalize()
    print(f"Today: {today.date()}")
    print(f"Gap between static history and today: {(today - history.index.max()).days} days\n")

    try:
        end = today.date()
        start = (history.index.max() - pd.Timedelta(days=7)).date()
        live_tail = prepare_meteostat_data(STATION_LAT, STATION_LON, start, end)
        forecast_input = build_live_forecast_input(history, live_tail)
        print(f"Live fetch OK. Blended input: {forecast_input.index.min().date()} to "
              f"{forecast_input.index.max().date()} ({len(forecast_input)} rows)\n")
    except Exception as exc:
        print(f"Live fetch failed ({exc}). Using cached history only.\n")
        base_cols = [c for c in BASE_WEATHER_COLUMNS if c in history.columns]
        forecast_input = history[base_cols]

    features = build_features(forecast_input.copy())
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    latest = features.iloc[[-1]].copy()

    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        print(f"WARNING - missing feature columns: {missing}\n")
    X = latest[feature_cols]

    print("=" * 60)
    print("INDIVIDUAL MODEL PREDICTIONS (not just the ensemble)")
    print("=" * 60)

    tmax_models = joblib.load(MODEL_DIR / "tmax_models.joblib")
    print("\n--- target_tmax ---")
    for name, model in tmax_models.items():
        try:
            pred = model.predict(X)[0]
            flag = "  <-- SUSPECT" if abs(pred) > 60 else ""
            print(f"  {name:22s}: {pred:10.2f} degC{flag}")
        except Exception as e:
            print(f"  {name:22s}: ERROR - {e}")

    tmin_models = joblib.load(MODEL_DIR / "tmin_models.joblib")
    print("\n--- target_tmin ---")
    for name, model in tmin_models.items():
        try:
            pred = model.predict(X)[0]
            flag = "  <-- SUSPECT" if abs(pred) > 60 else ""
            print(f"  {name:22s}: {pred:10.2f} degC{flag}")
        except Exception as e:
            print(f"  {name:22s}: ERROR - {e}")

    rain_models = joblib.load(MODEL_DIR / "rain_models.joblib")
    print("\n--- target_rain ---")
    for name, model in rain_models.items():
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else None
            proba_str = f"{proba * 100:.1f}%" if proba is not None else "N/A"
            print(f"  {name:22s}: class={pred}  P(rain)={proba_str}")
        except Exception as e:
            print(f"  {name:22s}: ERROR - {e}")

    print("\n" + "=" * 60)
    print("FEATURE ROW DIAGNOSTICS")
    print("=" * 60)
    print("Any NaN in the feature row?", bool(X.isna().any().any()))
    print("\nTop 10 largest-magnitude feature values (possible outliers vs. training):")
    print(X.iloc[0].abs().sort_values(ascending=False).head(10).to_string())

    print("\n" + "=" * 60)
    print("BACKTESTED TRAINING PERFORMANCE (training_results.json)")
    print("=" * 60)
    results_path = MODEL_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        print("\nRegression (MAE in degC, lower is better):")
        for name, m in results.get("regression", {}).items():
            print(f"  {name:30s} MAE={m['mae']:7.3f}  RMSE={m['rmse']:7.3f}  R2={m['r2']:7.3f}")
        print("\nClassification (rain):")
        for name, m in results.get("classification", {}).items():
            roc = m.get("roc_auc")
            roc_str = f"{roc:.3f}" if roc is not None else "N/A"
            print(f"  {name:22s} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  roc_auc={roc_str}")
    else:
        print("training_results.json not found - run src/train.py first.")


if __name__ == "__main__":
    main()