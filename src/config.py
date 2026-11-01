from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mahikeng_weather.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "weather_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_TMAX = "target_tmax"
TARGET_TMIN = "target_tmin"
TARGET_RAIN = "target_rain"

WEATHER_COLUMNS = [
    "t2m",
    "t2m_max",
    "t2m_min",
    "prectotcorr",
    "rh2m",
    "ws2m",
    "ps",
]

BASE_WEATHER_COLUMNS = [
    "t2m",
    "t2m_max",
    "t2m_min",
    "prectotcorr",
    "rh2m",
    "ws2m",
    "ps",
]

RAIN_THRESHOLD_MM = 0.1
ROLLING_WINDOWS = (3, 7, 14)
LAGS = (1, 2, 3, 7, 14)

# --- Meteostat integration (fallback / gap-filling source) ---
STATION_LAT = -25.864
STATION_LON = 25.644
STATION_SEARCH_RADIUS = 150_000
STATION_SEARCH_LIMIT = 5