"""
compare_sources.py

Benchmarks NASA POWER vs. meteostat vs. a combined (gap-filled) dataset on
the exact same feature pipeline, model set, and chronological train/test
split used by train.py — so the only thing that varies between runs is the
data source.

Usage:
    python -m src.compare_sources --data data/raw/mahikeng_weather.csv

Outputs:
    - Printed regression/classification metrics per source
    - models/source_comparison.csv  (one row per source x model x metric set)
"""

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    BASE_WEATHER_COLUMNS,
    MODEL_DIR,
    RAW_DATA_PATH,
    STATION_LAT,
    STATION_LON,
)
from .data import (
    clean_weather_data,
    last_valid_date,
    load_meteostat_daily,
    load_nasa_power_csv,
    merge_sources,
)
from .evaluate import classification_metrics, regression_metrics
from .features import finalise_training_data
from .models import classification_models, regression_models
from .train import time_split


def build_sources(data_path: str | Path, skiprows: int = 15) -> dict[str, pd.DataFrame]:
    """Build the three comparable, cleaned datasets from the same date range."""
    nasa_raw = load_nasa_power_csv(data_path, skiprows=skiprows)
    # Trim NASA POWER's trailing placeholder rows (see last_valid_date's docstring)
    # so meteostat is only asked for dates NASA POWER actually has real data for.
    end_date = last_valid_date(nasa_raw, BASE_WEATHER_COLUMNS)
    nasa_raw = nasa_raw.loc[:end_date]

    meteo_raw = load_meteostat_daily(
        lat=STATION_LAT,
        lon=STATION_LON,
        start=nasa_raw.index.min(),
        end=end_date,
    )

    combined = merge_sources(nasa_raw, meteo_raw)
    nasa_clean = clean_weather_data(nasa_raw)
    meteo_clean = clean_weather_data(meteo_raw)

    return {
        "nasa": nasa_clean,
        "meteostat": meteo_clean,
        "combined": combined,
    }


def evaluate_source(df: pd.DataFrame) -> dict:
    """Run the full feature/train/evaluate pipeline for one data source."""
    data, feature_cols = finalise_training_data(df)
    train, test = time_split(data, train_fraction=0.8)

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Not enough rows after feature engineering to split (got {len(data)} total). "
            "This usually means the source's date range or gaps were too small."
        )

    X_train, X_test = train[feature_cols], test[feature_cols]

    results = {"regression": {}, "classification": {}, "n_train": len(train), "n_test": len(test)}

    for target, key in [("target_tmax", "tmax"), ("target_tmin", "tmin")]:
        for name, model in regression_models().items():
            model.fit(X_train, train[target])
            predicted = model.predict(X_test)
            results["regression"][f"{key}_{name}"] = regression_metrics(test[target], predicted)

    for name, model in classification_models().items():
        model.fit(X_train, train["target_rain"].astype(int))
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results["classification"][name] = classification_metrics(
            test["target_rain"].astype(int), predicted, probability
        )

    return results


def results_to_frame(all_results: dict[str, dict]) -> pd.DataFrame:
    """Flatten {source: {regression: {...}, classification: {...}}} into a tidy table."""
    rows = []
    for source, results in all_results.items():
        for metric_key, metrics in results["regression"].items():
            rows.append({"source": source, "task": "regression", "model": metric_key, **metrics})
        for model_name, metrics in results["classification"].items():
            rows.append({"source": source, "task": "classification", "model": model_name, **metrics})
    return pd.DataFrame(rows)


def compare_sources(data_path=RAW_DATA_PATH, skiprows: int = 15, model_dir=MODEL_DIR) -> pd.DataFrame:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    sources = build_sources(data_path, skiprows=skiprows)

    all_results = {}
    for name, df in sources.items():
        print(f"\n--- Evaluating source: {name} ({len(df)} cleaned rows) ---")
        try:
            all_results[name] = evaluate_source(df)
        except ValueError as exc:
            print(f"Skipping {name}: {exc}")

    table = results_to_frame(all_results)
    out_path = model_dir / "source_comparison.csv"
    table.to_csv(out_path, index=False)

    print("\n=== Regression (lower mae/rmse is better) ===")
    print(table[table["task"] == "regression"].sort_values(["model", "mae"]).to_string(index=False))

    print("\n=== Classification (higher f1 is better) ===")
    print(table[table["task"] == "classification"].sort_values(["model", "f1"], ascending=[True, False]).to_string(index=False))

    print(f"\nSaved full comparison table to {out_path}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--skiprows", type=int, default=15)
    args = parser.parse_args()

    compare_sources(args.data, args.skiprows)