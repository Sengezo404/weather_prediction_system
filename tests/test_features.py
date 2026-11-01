import numpy as np
import pandas as pd

from src.features import build_features


def sample_data(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "t2m": np.linspace(20, 30, n),
            "t2m_max": np.linspace(25, 35, n),
            "t2m_min": np.linspace(15, 25, n),
            "prectotcorr": np.zeros(n),
            "rh2m": np.full(n, 50.0),
            "ws2m": np.full(n, 3.0),
            "ps": np.full(n, 900.0),
        },
        index=idx,
    )


def test_targets_are_next_day():
    df = sample_data()
    result = build_features(df)
    assert result.loc[result.index[0], "target_tmax"] == 25.256410256410255
    assert result.loc[result.index[0], "target_tmin"] == 15.256410256410255


def test_rolling_features_use_previous_days():
    df = sample_data()
    result = build_features(df)
    first_valid = result["t2m_max_roll_mean_3"].first_valid_index()
    assert first_valid is not None
    expected = df.loc[:first_valid - pd.Timedelta(days=1), "t2m_max"].tail(3).mean()
    assert np.isclose(result.loc[first_valid, "t2m_max_roll_mean_3"], expected)
