import numpy as np
import pandas as pd

from .config import LAGS, ROLLING_WINDOWS, RAIN_THRESHOLD_MM


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    day_of_year = df.index.dayofyear

    df["month"] = df.index.month
    df["day_of_year"] = day_of_year
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Cyclic encoding prevents December and January being treated as far apart.
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # shift(1) is critical: tomorrow's target must never influence
        # today's features through a rolling window.
        past = df[col].shift(1)

        for window in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{window}"] = past.rolling(window).mean()
            df[f"{col}_roll_std_{window}"] = past.rolling(window).std()

            if col == "prectotcorr":
                df[f"{col}_roll_sum_{window}"] = past.rolling(window).sum()

    return df


def add_seasonal_history_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Expanding historical averages using only observations available before each date."""
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # Shift before expanding/grouping so the current day's value is excluded.
        shifted = df[col].shift(1)

        month_avg = shifted.groupby(df.index.month).expanding().mean()
        month_avg = month_avg.reset_index(level=0, drop=True).reindex(df.index)

        doy_avg = shifted.groupby(df.index.dayofyear).expanding().mean()
        doy_avg = doy_avg.reset_index(level=0, drop=True).reindex(df.index)

        df[f"month_hist_avg_{col}"] = month_avg
        df[f"doy_hist_avg_{col}"] = doy_avg

    return df


def add_targets(
    df: pd.DataFrame,
    rain_threshold_mm: float = RAIN_THRESHOLD_MM,
) -> pd.DataFrame:
    df = df.copy()

    if "t2m_max" not in df.columns or "t2m_min" not in df.columns:
        raise ValueError("t2m_max and t2m_min are required to create temperature targets.")
    if "prectotcorr" not in df.columns:
        raise ValueError("prectotcorr is required to create the rain target.")

    df["target_tmax"] = df["t2m_max"].shift(-1)
    df["target_tmin"] = df["t2m_min"].shift(-1)
    next_day_rain = df["prectotcorr"].shift(-1)
    df["target_rain"] = (next_day_rain >= rain_threshold_mm).astype(float)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features and next-day targets."""
    df = df.copy().sort_index()

    weather_cols = [
        c for c in [
            "t2m",
            "t2m_max",
            "t2m_min",
            "prectotcorr",
            "rh2m",
            "ws2m",
            "ps",
        ]
        if c in df.columns
    ]

    # IMPORTANT:
    # Create targets immediately from the original observations.
    # This guarantees that target_tmax at date T is t2m_max at T+1.
    df = add_targets(df)

    # Feature engineering uses only today's and previous observations.
    df = add_calendar_features(df)
    df = add_lag_features(df, weather_cols)
    df = add_rolling_features(df, weather_cols)

    df = add_seasonal_history_features(
        df,
        [
            c
            for c in [
                "t2m_max",
                "t2m_min",
                "prectotcorr",
                "rh2m",
                "ws2m",
                "ps",
            ]
            if c in df.columns
        ],
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    target_cols = {"target_tmax", "target_tmin", "target_rain"}
    return [c for c in df.columns if c not in target_cols]


def finalise_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = build_features(df)

    feature_cols = get_feature_columns(df)

    # Target rows at the end have no next-day observation.
    target_cols = ["target_tmax", "target_tmin", "target_rain"]
    df = df.dropna(subset=target_cols)

    # Feature rows may be incomplete during the first lag/rolling windows.
    df = df.dropna(subset=feature_cols)

    return df, feature_cols

