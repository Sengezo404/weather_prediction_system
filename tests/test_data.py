import pandas as pd
import numpy as np

from src.data import clean_weather_data


def test_nasa_sentinel_is_not_kept_as_real_weather():
    idx = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {
            "t2m": [20.0, -999.0, 22.0],
            "t2m_max": [25.0, 26.0, 27.0],
            "t2m_min": [15.0, 16.0, 17.0],
        },
        index=idx,
    )
    cleaned = clean_weather_data(df)
    assert -999.0 not in cleaned["t2m"].values
    assert np.isfinite(cleaned["t2m"]).all()
