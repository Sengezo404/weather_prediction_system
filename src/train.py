import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, STATION_LAT, STATION_LON
from .data import get_source_data
from .evaluate import regression_metrics, classification_metrics
from .features import finalise_training_data
from .models import regression_models, classification_models


def time_split(df: pd.DataFrame, train_fraction: float = 0.8):
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def fit_and_evaluate(
    data_path=RAW_DATA_PATH,
    skiprows=15,
    model_dir=MODEL_DIR,
    source: str = "nasa",
):
    """
    Train the tmax/tmin/rain models on the chosen data source
    ("nasa", "meteostat", or "combined") and save them to model_dir.

    Use compare_sources.py first to decide which source to pass here.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = get_source_data(source, data_path, lat=STATION_LAT, lon=STATION_LON, skiprows=skiprows)
    data, feature_cols = finalise_training_data(raw)

    train, test = time_split(data, train_fraction=0.8)
    X_train, X_test = train[feature_cols], test[feature_cols]

    results = {"source": source, "regression": {}, "classification": {}}

    # Temperature models.
    for target, filename in [
        ("target_tmax", "tmax_models.joblib"),
        ("target_tmin", "tmin_models.joblib"),
    ]:
        trained = {}
        for name, model in regression_models().items():
            model.fit(X_train, train[target])
            predicted = model.predict(X_test)
            results["regression"][target + "_" + name] = regression_metrics(
                test[target], predicted
            )
            trained[name] = model

        joblib.dump(trained, model_dir / filename)

    # Rain model.
    rain_models = {}
    for name, model in classification_models().items():
        model.fit(X_train, train["target_rain"].astype(int))
        predicted = model.predict(X_test)

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_test)[:, 1]

        results["classification"][name] = classification_metrics(
            test["target_rain"].astype(int),
            predicted,
            probability,
        )
        rain_models[name] = model

    joblib.dump(rain_models, model_dir / "rain_models.joblib")

    joblib.dump(feature_cols, model_dir / "feature_columns.joblib")

    # Save processed data for inspection/app charts.
    data.to_csv(PROCESSED_DATA_PATH)

    with open(model_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Training complete (source={source}).")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--skiprows", type=int, default=15)
    parser.add_argument(
        "--source",
        choices=["nasa", "meteostat", "combined"],
        default="nasa",
        help="Which dataset to train on. Run compare_sources.py first to decide.",
    )
    args = parser.parse_args()

    fit_and_evaluate(args.data, args.skiprows, source=args.source)