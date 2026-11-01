from pathlib import Path
import joblib
import pandas as pd

from .config import MODEL_DIR
from .features import build_features


class WeatherPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.tmax_models = joblib.load(self.model_dir / "tmax_models.joblib")
        self.tmin_models = joblib.load(self.model_dir / "tmin_models.joblib")
        self.rain_models = joblib.load(self.model_dir / "rain_models.joblib")
        self.feature_columns = joblib.load(self.model_dir / "feature_columns.joblib")

    def predict_from_history(self, history: pd.DataFrame) -> dict:
        """
        Predict tomorrow using a dataframe containing observations through today.

        The dataframe must have a DatetimeIndex and the weather columns used by
        the training pipeline.
        """
        if history.empty:
            raise ValueError("History dataframe is empty.")

        features = build_features(history.copy())

        # Only the final row is used: it represents today's known information.
        latest = features.iloc[[-1]].copy()

        missing = [c for c in self.feature_columns if c not in latest.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = latest[self.feature_columns]

        tmax = float(self.tmax_models["ensemble"].predict(X)[0])
        tmin = float(self.tmin_models["ensemble"].predict(X)[0])

        rain_model = self.rain_models["hist_gradient_boosting"]
        rain_class = int(rain_model.predict(X)[0])
        rain_probability = float(rain_model.predict_proba(X)[0, 1])

        return {
            "forecast_date": (latest.index[0] + pd.Timedelta(days=1)).date().isoformat(),
            "predicted_max_temperature": round(tmax, 1),
            "predicted_min_temperature": round(tmin, 1),
            "rain_tomorrow": bool(rain_class),
            "rain_probability": round(rain_probability, 3),
        }
