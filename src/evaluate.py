import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def regression_metrics(actual, predicted) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(rmse),
        "r2": float(r2_score(actual, predicted)),
    }


def classification_metrics(actual, predicted, probability=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }

    if probability is not None and len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))

    return result


def compare_regression_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model"):
        metrics = regression_metrics(group["actual"], group["predicted"])
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows).sort_values("mae")


def compare_classification_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model"):
        probability = group["probability"] if "probability" in group else None
        metrics = classification_metrics(
            group["actual"],
            group["predicted"],
            probability,
        )
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)
