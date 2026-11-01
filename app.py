import hmac
import json
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.fused_predict import (
    FUSED_MODEL_PATH,
    FUSED_RESULTS_PATH,
    prepare_live_fused_history,
    predict_next_day,
)
from src.multi_horizon import predict_10_days

ROOT = Path(__file__).resolve().parent
FUSED_PATH = ROOT / "data" / "processed" / "fused_weather.csv"
MH_RESULTS_PATH = ROOT / "data" / "processed" / "multi_horizon_results.json"

# Thresholds used for the plain-language advice. Adjust to local conditions.
RAIN_THRESHOLD = 0.5      # probability at or above which rain is "likely"
SHOWER_THRESHOLD = 0.3    # probability at or above which spraying is discouraged
HOT_C = 32
EXTREME_HEAT_C = 35
FROST_C = 3
RETRAIN_AFTER_DAYS = 30   # admin banner turns amber after this many days

PUBLIC_PAGES = ["Dashboard", "Forecast", "Advice", "About"]
RESEARCHER_PAGES = ["Data", "Model Results"]  # only shown after admin sign-in
NAV_ICONS = {
    "Dashboard": ":material/home:",
    "Forecast": ":material/calendar_month:",
    "Advice": ":material/eco:",
    "About": ":material/info:",
    "Data": ":material/bar_chart:",
    "Model Results": ":material/psychology:",
}

st.set_page_config(
    page_title="TemoSense",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ********************************************************************
# Theme (light / dark)
# ********************************************************************
# Start in whatever theme the visitor's Streamlit/system setting uses, then
# let the toggle in the top bar take over.
if "dark_mode" not in st.session_state:
    try:
        st.session_state.dark_mode = st.context.theme.type == "dark"
    except Exception:
        st.session_state.dark_mode = False

LIGHT_VARS = """
--bg1:#e6f2df;--bg2:#f8fbf6;--panel:rgba(255,255,255,.92);--card:#ffffff;
--text:#0e2f1c;--muted:#4a6556;--border:rgba(20,83,45,.12);
--shadow:0 6px 20px rgba(20,83,45,.10);
--brand:#14532d;--brand2:#0f4223;--accent:#7ac943;--accent-text:#0b2e14;
--t-temp:#fdecec;--t-rain:#e7f1fd;--t-hum:#e3f5f9;--t-wind:#e8f0fb;
--active-bg:#e3f5d8;--active-border:#3f9b3a;
--adv-warn:#ffe6a8;--adv-good:#dcf3cf;--adv-info:#d8eafc;--adv-text:#10261a;
--wet:#e7f1fd;--strong:#14532d;
"""
DARK_VARS = """
--bg1:#050d08;--bg2:#0b1710;--panel:rgba(27,48,36,.96);--card:#1d3326;
--text:#f4faf5;--muted:#bcd0c2;--border:rgba(255,255,255,.20);
--shadow:0 6px 22px rgba(0,0,0,.55);
--brand:#17603a;--brand2:#0f4a2b;--accent:#8fe36a;--accent-text:#06210f;
--t-temp:#4d2a2e;--t-rain:#183552;--t-hum:#16424e;--t-wind:#1c3853;
--active-bg:#24472c;--active-border:#8fe36a;
--adv-warn:#7a5a0f;--adv-good:#25572c;--adv-info:#1b4c76;--adv-text:#ffffff;
--wet:#183552;--strong:#a6f07e;
"""

BASE_CSS = """
.stApp{background:linear-gradient(180deg,var(--bg1) 0%,var(--bg2) 55%);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--bg2)}
[data-testid="stSidebar"] *{color:var(--text)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"],
[data-testid="stMetricValue"],[data-testid="stMetricLabel"],
[data-testid="stWidgetLabel"] p,[data-testid="stAlertContainer"] p{color:var(--text)}
.stApp{overflow-x:hidden}
.block-container{max-width:min(1500px,100%) !important;padding-top:3.2rem !important;padding-bottom:1rem !important;
 padding-left:clamp(.8rem,2.5vw,2.5rem) !important;padding-right:clamp(.8rem,2.5vw,2.5rem) !important}
[data-testid="stVerticalBlock"]{gap:.7rem}

/* ---------- top bar ---------- */
.st-key-topbar{background:linear-gradient(90deg,var(--brand),var(--brand2));border-radius:22px;
 padding:.7rem 1.3rem;box-shadow:var(--shadow);margin-bottom:1rem}
.st-key-topbar [data-testid="stHorizontalBlock"]{align-items:center}
.st-key-topbar label,.st-key-topbar label *{color:#fff !important}
.brand{display:flex;align-items:center;gap:.8rem}
.brand svg{flex:none}
.brand-name{font-size:1.9rem;font-weight:800;line-height:1.05;color:#fff;letter-spacing:-.01em}
.brand-sub{font-size:.85rem;color:rgba(255,255,255,.85)}
.st-key-nav{flex-wrap:wrap;row-gap:.3rem}
.st-key-nav button{background:transparent !important;border:1px solid transparent !important;border-radius:999px !important;
 padding:.45rem 1.05rem !important;min-height:0 !important;box-shadow:none !important;color:#fff !important;transition:background .15s}
.st-key-nav button *{color:inherit !important;font-weight:600;font-size:1rem}
.st-key-nav button:hover{background:rgba(255,255,255,.14) !important}
[class*="st-key-nav_on_"] button{background:var(--accent) !important;color:var(--accent-text) !important}
[class*="st-key-nav_on_"] button:hover{background:var(--accent) !important}

/* ---------- location row ---------- */
.loc{display:flex;align-items:center;gap:.7rem}
.loc-pin{font-size:2rem;line-height:1}
.loc-name{font-size:1.4rem;font-weight:800;color:var(--text);line-height:1.15}
.loc-sub{font-size:1rem;color:var(--muted)}
.inline-label{text-align:right;font-weight:600;color:var(--text);padding-top:.35rem}

/* ---------- metric tiles ---------- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:.8rem;margin:.3rem 0 .8rem}
.tile{display:flex;align-items:center;gap:.9rem;border-radius:20px;padding:.75rem 1.1rem;
 border:1px solid var(--border);box-shadow:var(--shadow);color:var(--text);min-width:0}
.tile.temp{background:var(--t-temp)} .tile.rain{background:var(--t-rain)}
.tile.hum{background:var(--t-hum)} .tile.wind{background:var(--t-wind)}
.tile .t-icon{font-size:2.3rem;line-height:1}
.tile .t-label{font-weight:600;font-size:1rem}
.tile .t-value{font-size:2.2rem;font-weight:800;line-height:1.1;color:var(--strong)}
.tile .t-value small{font-size:1.2rem;font-weight:700}
.tile .t-sub{font-size:.9rem;color:var(--muted)}

/* ---------- forecast panel ---------- */
.panel{background:var(--panel);border-radius:24px;padding:.9rem 1.1rem 1.1rem;margin-bottom:.8rem;
 border:1px solid var(--border);box-shadow:var(--shadow)}
.panel-head{display:flex;align-items:center;gap:.6rem;font-size:1.4rem;font-weight:800;color:var(--text);margin-bottom:.8rem}
.tf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(150px,45%),1fr));gap:.8rem}
.tf-card{background:var(--card);border:2px solid transparent;border-radius:18px;padding:.6rem .5rem;
 text-align:center;min-width:0;color:var(--text);box-shadow:0 2px 10px rgba(0,0,0,.06)}
.tf-card.active{background:var(--active-bg);border-color:var(--active-border)}
.tf-card.wet .tf-icon{filter:none}
.tf-card.far .tf-temp,.tf-card.far .tf-icon{opacity:.8}
.tf-label{font-weight:800;font-size:1.1rem}
.tf-date{font-size:.9rem;color:var(--muted)}
.tf-icon{font-size:clamp(2rem,5vh,2.8rem);line-height:1.2}
.tf-temp{font-size:clamp(1.5rem,3.8vh,2rem);font-weight:800;color:var(--strong);line-height:1.1}
.tf-min{font-size:.85rem;color:var(--muted)}
.tf-rain{font-size:1.05rem;font-weight:600;margin-top:.1rem}
.tf-bar{height:.4rem;border-radius:.2rem;background:rgba(128,128,128,.22);margin:.45rem .6rem .3rem}
.tf-bar span{display:block;height:100%;border-radius:.2rem;background:#3b82f6}

/* ---------- advice band ---------- */
.band{background:linear-gradient(90deg,var(--brand),var(--brand2));border-radius:24px;padding:1rem 1.1rem 1.1rem;
 margin-bottom:1rem;box-shadow:var(--shadow)}
.band-head{display:flex;align-items:center;gap:.6rem;font-size:1.35rem;font-weight:800;color:#fff;margin:.1rem 0 .8rem .3rem}
.adv-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:.8rem}
.adv{display:flex;gap:.8rem;border-radius:16px;padding:.9rem 1rem;color:var(--adv-text)}
.adv.warn{background:var(--adv-warn)} .adv.good{background:var(--adv-good)} .adv.info{background:var(--adv-info)}
.adv-icon{font-size:2rem;line-height:1.1}
.adv-title{font-weight:800;font-size:1.1rem}
.adv-text{font-size:.95rem;line-height:1.35}

/* ---------- widgets follow the chosen theme ---------- */
[data-baseweb="select"] > div,[data-baseweb="input"],[data-baseweb="input"] > div{background:var(--card) !important;border-color:var(--border) !important}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:var(--text) !important}
[data-baseweb="select"] svg{fill:var(--text) !important}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li,[data-baseweb="menu"]{background:var(--card) !important;color:var(--text) !important}
[data-baseweb="popover"] li *{color:var(--text) !important}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{background:var(--active-bg) !important}
[data-testid="stAlertContainer"]{background:var(--card) !important;border:1px solid var(--border);border-radius:14px}
[data-testid="stAlertContainer"] *{color:var(--text) !important}
[data-testid="stExpander"] details{background:var(--card);border-color:var(--border)}
[data-testid="stExpander"] summary *{color:var(--text) !important}
.stApp hr{border-color:var(--border)}
.stDownloadButton button,[data-testid="stSidebar"] .stButton button,[data-testid="stFormSubmitButton"] button{
 background:var(--card) !important;border:1px solid var(--border) !important;color:var(--text) !important}
.stDownloadButton button *,[data-testid="stSidebar"] .stButton button *,[data-testid="stFormSubmitButton"] button *{color:var(--text) !important}
code,pre{color:var(--text) !important}

/* ---------- misc ---------- */
.badge{display:inline-block;padding:.2rem .7rem;border-radius:8px;font-size:.8rem;margin:0 .4rem .4rem 0;
 border:1px solid var(--border);color:var(--text)}
.badge.ok{background:rgba(34,197,94,.18)} .badge.warn{background:rgba(245,158,11,.22)} .badge.bad{background:rgba(239,68,68,.18)}
img,svg,canvas{max-width:100%}
@media (max-width:900px){
 [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;gap:.6rem !important}
 [data-testid="stColumn"],[data-testid="column"]{flex:1 1 240px !important;min-width:min(100%,240px) !important;width:auto !important}
 .brand-name{font-size:1.6rem}
 .inline-label{text-align:left}
}
@media (max-width:700px){
 .st-key-topbar{padding:.6rem .8rem;border-radius:18px}
 .brand-sub{display:none}
 .brand svg{width:40px;height:40px}
 .st-key-nav button{padding:.4rem .75rem !important}
 .st-key-nav button *{font-size:.92rem}
 .loc-name{font-size:1.2rem}
}
@media (max-width:480px){
 .tile{padding:.8rem} .tile .t-value{font-size:1.8rem}
 .panel{padding:.9rem}
}
"""


def inject_css() -> None:
    dark = bool(st.session_state.get("dark_mode"))
    try:
        native_dark = st.context.theme.type == "dark"
    except Exception:
        native_dark = False
    vars_block = DARK_VARS if dark else LIGHT_VARS
    # Streamlit tables and charts draw in Streamlit's own theme. When that differs
    # from the chosen theme, invert them so they stay readable.
    fix = ""
    if dark != native_dark:
        fix = (
            '[data-testid="stDataFrame"],[data-testid="stVegaLiteChart"],'
            '[data-testid="stArrowVegaLiteChart"]{filter:invert(1) hue-rotate(180deg)}'
        )
    st.markdown(
        f"<style>:root{{{vars_block}}}{BASE_CSS}{fix}</style>",
        unsafe_allow_html=True,
    )


inject_css()


def html(markup: str) -> None:
    """Render HTML without markdown mistaking indentation for a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ********************************************************************
# Admin login
# ********************************************************************
def admin_login() -> bool:
    st.sidebar.markdown("### Researcher access")
    if st.session_state.get("admin_logged_in", False):
        if st.sidebar.button("Log out"):
            st.session_state.admin_logged_in = False
            st.rerun()
        return True

    if st.session_state.get("failed_attempts", 0) >= 5:
        st.sidebar.error("Too many attempts. Reload the page to try again.")
        return False

    with st.sidebar.form("admin_login_form"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        try:
            configured = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            configured = ""
        if configured and hmac.compare_digest(password, str(configured)):
            st.session_state.admin_logged_in = True
            st.session_state.failed_attempts = 0
            st.rerun()
        st.session_state.failed_attempts = st.session_state.get("failed_attempts", 0) + 1
        st.sidebar.error("Incorrect password or admin password not configured.")
    return False


is_admin = admin_login()


# ********************************************************************
# Data loading (cached so widget clicks do not refetch or re-predict)
# ********************************************************************
def load_results():
    if not FUSED_RESULTS_PATH.exists():
        return None
    return json.loads(FUSED_RESULTS_PATH.read_text(encoding="utf-8"))


def load_mh_results():
    if not MH_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(MH_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=1800, show_spinner="Fetching the latest weather data…")
def load_live():
    return prepare_live_fused_history(days=60)


@st.cache_data
def load_history(path_str: str, modified: float):
    return pd.read_csv(path_str, index_col=0, parse_dates=True)


@st.cache_data(ttl=1800)
def cached_forecast(frame: pd.DataFrame):
    return predict_next_day(frame)


def build_outlook(frame: pd.DataFrame, forecast_day: pd.Timestamp, days: int = 10) -> pd.DataFrame:
    """Return up to `days` rows strictly after the forecast day, with clean columns."""
    raw = predict_10_days(frame).copy()
    raw = raw.rename(
        columns={
            "date": "Date",
            "predicted_max_temperature": "Maximum Temperature",
            "predicted_min_temperature": "Minimum Temperature",
            "rain_probability": "Rain Probability",
        }
    )
    if "Date" in raw.columns:
        dates = pd.to_datetime(raw["Date"]).dt.normalize()
    else:
        dates = pd.to_datetime([forecast_day + pd.Timedelta(days=i) for i in range(len(raw))])

    rain = pd.to_numeric(raw["Rain Probability"], errors="coerce")
    rain = rain.where(rain <= 1, rain / 100)  # accept 0-1 or 0-100

    out = pd.DataFrame(
        {
            "date": pd.Series(dates).values,
            "tmax": pd.to_numeric(raw["Maximum Temperature"], errors="coerce").values,
            "tmin": pd.to_numeric(raw["Minimum Temperature"], errors="coerce").values,
            "rain": rain.values,
            "horizon": (
                pd.to_numeric(raw["Horizon"], errors="coerce").values
                if "Horizon" in raw.columns
                else np.nan
            ),
        }
    ).dropna(subset=["date", "tmax", "tmin", "rain"])
    out = out[out["date"] > forecast_day].sort_values("date")
    return out.head(days).reset_index(drop=True)


@st.cache_data(ttl=1800)
def cached_outlook(frame: pd.DataFrame, forecast_day: pd.Timestamp):
    return build_outlook(frame, forecast_day)

# Download the fused model automatically on Streamlit Cloud
import os
import requests

MODEL_URL = (
    "https://github.com/Sengezo404/weather_prediction_system/"
    "releases/download/v1.0-model/fused_models_compressed9.joblib"
)

if not FUSED_MODEL_PATH.exists():
    FUSED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = FUSED_MODEL_PATH.with_suffix(".download")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=120) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        os.replace(temp_path, FUSED_MODEL_PATH)

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        st.error(f"Could not download the fused prediction model: {e}")
        st.stop()
# ********************************************************************
# Required model/data checks
# ------------------------------------------------------------------
if not FUSED_MODEL_PATH.exists() or not FUSED_PATH.exists():
    if is_admin:
        st.error("The fused model or dataset has not been created yet.")
        st.code("python -m src.fused_system", language="powershell")
    else:
        st.info("The forecast is being prepared. Please check back soon.")
    st.stop()

history = load_history(str(FUSED_PATH), FUSED_PATH.stat().st_mtime)

try:
    live_history, live_meta = load_live()
    forecast = cached_forecast(live_history)
    forecast_input = live_history
    data_mode = "live"
except Exception as exc:
    print(f"LIVE WEATHER ERROR: {type(exc).__name__}: {exc}", flush=True)
    live_history = history
    forecast_input = history
    try:
        forecast = cached_forecast(history)
        data_mode = "cached"
        live_meta = {"error": str(exc)}
    except Exception as fallback_exc:
        if is_admin:
            st.error(f"Prediction failed: {fallback_exc}")
        else:
            st.info("The forecast is temporarily unavailable. Please try again later.")
        st.stop()

results = load_results()
mh_results = load_mh_results()

forecast_date = forecast.get("forecast_date")
if forecast_date:
    forecast_day = pd.Timestamp(forecast_date).normalize()
else:
    forecast_day = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
today = pd.Timestamp.today().normalize()

try:
    outlook = cached_outlook(forecast_input, forecast_day)
    outlook_error = None
except Exception as exc:
    outlook = pd.DataFrame(columns=["date", "tmax", "tmin", "rain", "horizon"])
    outlook_error = str(exc)

# One table with the forecast day first, then the days after it.
full_forecast = pd.concat(
    [
        pd.DataFrame(
            {
                "date": [forecast_day],
                "tmax": [forecast["predicted_max_temperature"]],
                "tmin": [forecast["predicted_min_temperature"]],
                "rain": [forecast["rain_probability"]],
                "horizon": [np.nan],
            }
        ),
        outlook,
    ],
    ignore_index=True,
)


# ********************************************************************
# Plain-language helpers
# ------------------------------------------------------------------
def day_icon(rain: float, tmax: float, tmin: float) -> str:
    if rain >= RAIN_THRESHOLD:
        return "🌧️"
    if rain >= SHOWER_THRESHOLD:
        return "☁️"
    if tmin <= FROST_C:
        return "❄️"
    if rain >= 0.15:
        return "⛅"
    return "☀️"


def heat_level(tmax: float) -> str:
    if tmax >= EXTREME_HEAT_C:
        return "Very hot"
    if tmax >= HOT_C:
        return "Hot"
    if tmax >= 25:
        return "Warm"
    if tmax >= 15:
        return "Mild"
    return "Cool"


def rain_note(p: float) -> str:
    if p >= RAIN_THRESHOLD:
        return "Rain likely."
    if p >= SHOWER_THRESHOLD:
        return "Showers possible."
    return "Rain unlikely."


def day_label(i: int) -> str:
    if i == 0:
        if forecast_day == today:
            return "Today"
        if forecast_day == today + pd.Timedelta(days=1):
            return "Tomorrow"
        return "Day 1"
    return f"Day {i + 1}"


def obs_value(options):
    """Latest observed value for the first available column, or None."""
    last = forecast_input.iloc[-1]
    for col in options:
        if col in forecast_input.columns and pd.notna(last[col]):
            return float(last[col])
    return None


def build_advice(fc: dict, days: pd.DataFrame):
    """Return advice as (emoji, title, text, tone) with tone in warn/good/info."""
    p = fc["rain_probability"]
    tmax = fc["predicted_max_temperature"]
    tmin = fc["predicted_min_temperature"]

    next_rain = None
    if not days.empty:
        wet = days[days["rain"] >= RAIN_THRESHOLD]
        if not wet.empty:
            next_rain = wet.iloc[0]["date"]

    items = []
    if tmin <= FROST_C:
        items.append(("❄️", "Frost risk", "A cold night is forecast. Protect seedlings and frost-sensitive crops.", "warn"))
    if tmax >= EXTREME_HEAT_C:
        items.append(("🌡️", "Heat stress risk", "Extreme heat is forecast. Avoid strenuous work in the afternoon, and water livestock early and provide shade.", "warn"))
    elif tmax >= HOT_C:
        items.append(("🌡️", "Heat stress risk", "Water livestock early and provide shade.", "warn"))

    if p >= SHOWER_THRESHOLD:
        items.append(("🧴", "Avoid spraying", "Rain could wash the product off. Wait for a drier day.", "warn"))
    else:
        until = f" until {next_rain:%A}" if next_rain is not None else ""
        items.append(("🧴", "Good spraying window", f"No rain expected{until}. This forecast does not include wind, so check it locally.", "good"))

    if p >= RAIN_THRESHOLD:
        irrigation = ("💧", "Hold off irrigating", "Rain is likely. Check soil moisture afterwards.", "info")
    elif tmax >= 30:
        irrigation = ("💧", "Irrigate in the evening", "Less water lost to evaporation.", "info")
    else:
        irrigation = ("💧", "Check soil moisture", "Check soil moisture before deciding to irrigate.", "info")
    if p < RAIN_THRESHOLD and next_rain is not None:
        irrigation = (irrigation[0], irrigation[1], irrigation[2] + f" Rain looks more likely on {next_rain:%A}.", irrigation[3])
    items.append(irrigation)

    if p >= RAIN_THRESHOLD:
        items.append(("🚜", "Field work", "Fields may be wet or muddy. Keep heavy machinery off waterlogged soil.", "warn"))
    elif tmax >= HOT_C:
        items.append(("🚜", "Field work", "Work early and rest at midday. Make sure workers have shade and water.", "warn"))
    else:
        items.append(("🚜", "Field work", "Good conditions for field work.", "good"))
    return items


def advice_html(items, title: str) -> str:
    cells = "".join(
        f"""
        <div class="adv {tone}">
          <div class="adv-icon">{emoji}</div>
          <div><div class="adv-title">{head}</div><div class="adv-text">{text}</div></div>
        </div>
        """
        for emoji, head, text, tone in items
    )
    return f'<div class="band"><div class="band-head">🌿 {title}</div><div class="adv-grid">{cells}</div></div>'


def forecast_panel_html(rows: pd.DataFrame, title: str) -> str:
    cards = []
    for i, r in enumerate(rows.itertuples()):
        cls = "tf-card"
        if i == 0:
            cls += " active"
        if r.rain >= RAIN_THRESHOLD:
            cls += " wet"
        if i >= 5:
            cls += " far"
        cards.append(
            f"""
            <div class="{cls}">
              <div class="tf-label">{day_label(i)}</div>
              <div class="tf-date">{r.date:%a, %d %b}</div>
              <div class="tf-icon">{day_icon(r.rain, r.tmax, r.tmin)}</div>
              <div class="tf-temp">{r.tmax:.0f}°C</div>
              <div class="tf-min">Low {r.tmin:.0f}°C</div>
              <div class="tf-bar"><span style="width:{min(max(r.rain, 0), 1) * 100:.0f}%"></span></div>
              <div class="tf-rain">{r.rain * 100:.0f}% rain</div>
            </div>
            """
        )
    return (
        f'<div class="panel"><div class="panel-head">📅 {title}</div>'
        f'<div class="tf-grid">{"".join(cards)}</div></div>'
    )


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%Y-%m-%d"),
            "Day": df["date"].dt.strftime("%a"),
            "Maximum": df["tmax"].map(lambda v: f"{v:.1f} °C"),
            "Minimum": df["tmin"].map(lambda v: f"{v:.1f} °C"),
            "Rain probability": df["rain"].map(lambda v: f"{v * 100:.0f}%"),
            "Rain expected": df["rain"].map(lambda v: "Yes 🌧️" if v >= RAIN_THRESHOLD else "No ☀️"),
        }
    )


# ********************************************************************
# Top bar: brand, navigation, light/dark toggle
# ********************************************************************
LOGO = """
<svg width="52" height="52" viewBox="0 0 64 64" aria-hidden="true">
  <path d="M30 60C30 40 22 30 4 25c0 20 9 31 26 35z" fill="#7ac943"/>
  <path d="M32 48c0-22 9-37 28-44 2 22-7 40-28 44z" fill="#b5e88c"/>
  <path d="M31 62V28" stroke="#e4f8d3" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>
"""


def render_topbar() -> str:
    pages = PUBLIC_PAGES + (RESEARCHER_PAGES if is_admin else [])
    st.session_state.setdefault("page", "Dashboard")
    if st.session_state.page not in pages:  # e.g. after a researcher logs out
        st.session_state.page = "Dashboard"

    def go(name: str) -> None:
        st.session_state.page = name

    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([3, 7, 1.7], vertical_alignment="center")
        with c1:
            html(
                f"""
                <div class="brand">{LOGO}
                  <div><div class="brand-name">TemoSense</div>
                  <div class="brand-sub">Smart Weather Prediction for Farmers</div></div>
                </div>
                """
            )
        with c2:
            with st.container(key="nav", horizontal=True, horizontal_alignment="center", gap="small"):
                for name in pages:
                    on = name == st.session_state.page
                    st.button(
                        name,
                        key=f"nav_{'on' if on else 'off'}_{name.replace(' ', '_')}",
                        icon=NAV_ICONS[name],
                        on_click=go,
                        args=(name,),
                    )
        with c3:
            st.toggle(":material/dark_mode: Dark mode", key="dark_mode")
    return st.session_state.page


# ********************************************************************
# Pages
# ********************************************************************
def render_dashboard():
    if data_mode == "cached":
        st.warning("Recent weather updates are unavailable, so this forecast may be out of date.")

    c_loc, c_lab, c_sel = st.columns([6, 1.3, 1.7], vertical_alignment="center")
    with c_loc:
        html(
            """
            <div class="loc"><span class="loc-pin">📍</span>
              <div><div class="loc-name">Mahikeng, North West</div>
              <div class="loc-sub">Local agricultural weather outlook</div></div>
            </div>
            """
        )
    with c_lab:
        html('<div class="inline-label">Forecast:</div>')
    with c_sel:
        choices = [3, 5, 7, 10]
        n = st.selectbox(
            "Forecast length",
            choices,
            index=0,
            format_func=lambda d: f"{d} days",
            key="horizon_days",
            label_visibility="collapsed",
        )

    days = full_forecast.head(n)
    shown = len(days)
    tmax = forecast["predicted_max_temperature"]
    rain = forecast["rain_probability"]

    humidity = obs_value(["fused_rh2m", "rh2m"])
    wind = obs_value(["fused_ws2m", "ws2m"])  # m/s in the model inputs
    obs_date = f"{pd.Timestamp(forecast_input.index[-1]):%d %b}"
    hum_text = "N/A" if humidity is None else f"{humidity:.0f}%"
    wind_text = "N/A" if wind is None else f"{wind * 3.6:.0f} <small>km/h</small>"

    html(
        f"""
        <div class="tiles">
          <div class="tile temp"><span class="t-icon">🌡️</span>
            <div><div class="t-label">Temperature</div><div class="t-value">{tmax:.0f}°C</div>
            <div class="t-sub">Predicted maximum.</div></div></div>
          <div class="tile rain"><span class="t-icon">🌧️</span>
            <div><div class="t-label">Rain probability</div><div class="t-value">{rain * 100:.0f}%</div>
            <div class="t-sub">{rain_note(rain)}</div></div></div>
          <div class="tile hum"><span class="t-icon">💧</span>
            <div><div class="t-label">Humidity</div><div class="t-value">{hum_text}</div>
            <div class="t-sub">Observed {obs_date}.</div></div></div>
          <div class="tile wind"><span class="t-icon">🌬️</span>
            <div><div class="t-label">Wind speed</div><div class="t-value">{wind_text}</div>
            <div class="t-sub">Observed {obs_date}.</div></div></div>
        </div>
        """
    )

    if days.empty:
        st.info("The forecast is temporarily unavailable.")
    else:
        html(forecast_panel_html(days, f"{shown}-Day Weather Forecast"))
        if shown < n:
            st.caption(f"Only {shown} days are available right now.")

    items = build_advice(forecast, days.iloc[1:])[:3]
    html(advice_html(items, f"Farm advice for the next {shown} days"))
    st.caption(
        "These are model predictions, not guarantees. Check local conditions and soil "
        "moisture before big decisions. Accuracy drops for days further ahead."
    )


def render_forecast_page():
    st.subheader("Forecast")
    if outlook_error:
        if is_admin:
            st.error(f"Multi-horizon forecast could not be loaded: {outlook_error}")
            st.code("python -m src.multi_horizon", language="powershell")
        else:
            st.info("The extended forecast is temporarily unavailable.")
    html(forecast_panel_html(full_forecast, f"{len(full_forecast)}-Day Weather Forecast"))
    st.dataframe(format_table(full_forecast), width="stretch", hide_index=True)
    st.caption("Fainter days are less certain. Accuracy falls the further ahead you look.")

    if is_admin:
        st.download_button(
            "Download forecast (CSV)",
            full_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"mahikeng_forecast_{forecast_day:%Y%m%d}.csv",
            mime="text/csv",
        )
        with st.expander("Show chart"):
            if mh_results:
                st.caption("Red: maximum. Blue: minimum. Shaded band: middle 80% of past errors at that horizon.")
            st.altair_chart(band_chart(full_forecast, mh_results))
            st.bar_chart(
                full_forecast.set_index("date")[["rain"]].rename(columns={"rain": "Rain probability"}),
                width="stretch",
            )


def render_advice_page():
    st.subheader("Farm advice")
    items = build_advice(forecast, outlook)
    html(advice_html(items, f"Farm advice for the next {len(full_forecast)} days"))
    st.caption(
        "Advice is based on the forecast for "
        f"{forecast_day:%A %d %B} and the days after it. Check local conditions before big decisions."
    )


def band_chart(days: pd.DataFrame, mh):
    """Max/min temperature lines with a band covering the middle 80% of test errors."""
    reg = (mh or {}).get("regression", {})
    d = days.copy()

    def shift(h, kind, key):
        if pd.isna(h):
            return 0.0
        return float(reg.get(f"day_{int(h)}_{kind}", {}).get(key) or 0.0)

    for kind in ("max", "min"):
        d[f"{kind}_lo"] = d[f"t{kind}"] + [shift(h, kind, "q10") for h in d["horizon"]]
        d[f"{kind}_hi"] = d[f"t{kind}"] + [shift(h, kind, "q90") for h in d["horizon"]]

    base = alt.Chart(d).encode(x=alt.X("date:T", title=None))
    layers = []
    for kind, colour in (("max", "#ef4444"), ("min", "#3b82f6")):
        layers.append(
            base.mark_area(opacity=0.2, color=colour).encode(
                y=alt.Y(f"{kind}_lo:Q", title="°C", scale=alt.Scale(zero=False)),
                y2=f"{kind}_hi:Q",
            )
        )
        layers.append(base.mark_line(color=colour, point=True).encode(y=f"t{kind}:Q"))
    return alt.layer(*layers).properties(width="container", height=280)


def model_age_days():
    try:
        modified = datetime.fromtimestamp(FUSED_MODEL_PATH.stat().st_mtime)
        return (datetime.now() - modified).days, modified
    except OSError:
        return None, None


def badge(text: str, level: str) -> str:
    return f'<span class="badge {level}">{text}</span>'


def render_data_page():
    st.subheader("Data")
    latest = forecast_input.iloc[-1]
    st.caption(
        f"Latest observation: {pd.Timestamp(forecast_input.index[-1]):%d %B %Y}. "
        "These are model inputs, not forecast values."
    )
    weather = [
        ("Temperature", ["fused_t2m", "t2m"], "°C"),
        ("Maximum temperature", ["fused_t2m_max", "t2m_max"], "°C"),
        ("Minimum temperature", ["fused_t2m_min", "t2m_min"], "°C"),
        ("Rainfall", ["fused_prectotcorr", "prectotcorr"], "mm"),
        ("Relative humidity", ["fused_rh2m", "rh2m"], "%"),
        ("Wind speed", ["fused_ws2m", "ws2m"], "m/s"),
        ("Surface pressure", ["fused_ps", "ps"], "hPa"),
    ]
    cols = st.columns(4)
    shown = 0
    for label, options, unit in weather:
        col = next((c for c in options if c in forecast_input.columns), None)
        if col is None:
            continue
        value = latest[col]
        text = "N/A" if pd.isna(value) else f"{float(value):.1f} {unit}"
        cols[shown % 4].metric(label, text)
        shown += 1
    if shown == 0:
        st.info("No observation columns are available in the current prediction input.")

    if not is_admin:
        st.info("History charts and data health are available to signed-in researchers.")
        return

    st.divider()
    age_days, trained_at = model_age_days()
    badges = [
        badge("Live input from both sources", "ok")
        if data_mode == "live"
        else badge("Cached data, live update failed", "warn")
    ]
    if age_days is not None:
        level = "warn" if age_days > RETRAIN_AFTER_DAYS else "ok"
        badges.append(badge(f"Model trained {age_days} days ago", level))
    html("".join(badges))

    st.subheader("Fused historical weather")
    max_days = min(180, len(live_history))
    if max_days >= 7:
        days = st.slider("Days to display", 7, max_days, min(30, max_days))
    else:
        days = max_days
        st.info(f"Only {max_days} observations are available.")
    recent = live_history.tail(days)

    temp_cols = [c for c in ["fused_t2m_max", "fused_t2m_min"] if c in recent.columns]
    if temp_cols:
        st.line_chart(recent[temp_cols], width="stretch")
    if "fused_prectotcorr" in recent.columns:
        st.bar_chart(recent[["fused_prectotcorr"]], width="stretch")
    if all(c in recent.columns for c in ["fused_rh2m", "fused_ws2m"]):
        st.line_chart(recent[["fused_rh2m", "fused_ws2m"]], width="stretch")

    fused_cols = [c for c in recent.columns if c.startswith("fused_")]
    if fused_cols:
        st.dataframe(recent[fused_cols].round(2), width="stretch")
        st.download_button(
            "Download displayed history (CSV)",
            recent[fused_cols].round(3).to_csv().encode("utf-8"),
            file_name="mahikeng_fused_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No fused columns are available to display.")

    st.divider()
    st.subheader("Data health")
    h1, h2, h3 = st.columns(3)
    h1.metric("NASA POWER through", str(live_meta.get("nasa_through", "n/a")))
    h2.metric("Meteostat through", str(live_meta.get("meteostat_through", "n/a")))
    if fused_cols:
        missing = live_history.tail(60)[fused_cols].isna().mean().mean() * 100
        h3.metric("Missing values (last 60 days)", f"{missing:.1f}%")
    if trained_at is not None:
        st.write(f"**Last model file update:** {trained_at:%d %b %Y %H:%M}")
    if data_mode == "cached" and "error" in live_meta:
        st.info(f"Live update note: {live_meta['error']}")


def render_horizon_table(mh):
    st.write("### Accuracy by forecast horizon")
    if not mh:
        st.info("No per-horizon results yet. Run `python -m src.multi_horizon`.")
        return
    reg = mh.get("regression", {})
    clf = mh.get("classification", {})
    hours = sorted({int(k.split("_")[1]) for k in list(reg) + list(clf)})
    rows = []
    for h in hours:
        mx = reg.get(f"day_{h}_max", {})
        mn = reg.get(f"day_{h}_min", {})
        rn = clf.get(f"day_{h}_rain", {})
        rows.append(
            {
                "Day": h,
                "Max MAE (°C)": mx.get("mae"),
                "Max skill": mx.get("skill"),
                "Min MAE (°C)": mn.get("mae"),
                "Min skill": mn.get("skill"),
                "Rain F1": rn.get("f1"),
                "Rain AUC": rn.get("roc_auc"),
                "Rain Brier": rn.get("brier"),
                "Rain skill": rn.get("brier_skill"),
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table.round(3), width="stretch", hide_index=True)
    st.line_chart(table.set_index("Day")[["Max MAE (°C)", "Min MAE (°C)"]], width="stretch")
    st.caption(
        "Skill compares the model with a simple baseline: 0 means no better, 1 means perfect, "
        "negative means worse. Temperature baseline: today's value repeats. Rain baseline: the "
        "training-period rain frequency (Brier skill score). Day 1 is one day after the latest "
        "observation. The shaded band on the forecast chart covers the middle 80% of test errors."
    )


def render_model_page():
    st.subheader("Model results")
    if not is_admin:
        st.info("Model results are available to signed-in researchers. Use the sidebar to sign in.")
        return
    if results:
        regression = [
            {"Model": n, "MAE": m.get("mae"), "RMSE": m.get("rmse"), "R²": m.get("r2")}
            for n, m in results.get("regression", {}).items()
        ]
        if regression:
            st.write("### Regression")
            st.dataframe(pd.DataFrame(regression).round(4), width="stretch", hide_index=True)

        classification = [
            {
                "Model": n,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
            }
            for n, m in results.get("classification", {}).items()
        ]
        if classification:
            st.write("### Rain classification")
            st.dataframe(pd.DataFrame(classification).round(4), width="stretch", hide_index=True)

        test_start = results.get("data", {}).get("test_start")
        if test_start:
            st.caption(
                f"Chronological test period begins {test_start}. "
                "No future observations were used to train the models."
            )
        else:
            st.caption("Models were evaluated on a chronological holdout set.")
    else:
        st.info("No model-performance results file is available yet.")
    st.divider()
    render_horizon_table(mh_results)


def render_about_page():
    st.subheader("About TemoSense")
    st.markdown(
        """
        **Purpose:** Mahikeng weather prediction for research and smart-agriculture decision support.

        **Training data:** NASA POWER and Meteostat historical weather.

        **Live input:** the latest observations from both sources are fetched, harmonised to
        common units, fused by date, and passed through the same feature pipeline used in training.

        **Predictions:** maximum and minimum temperature, rain occurrence and rain probability,
        for the next day and ten days ahead using separately trained horizon models.

        **Note:** NASA POWER and Meteostat are not treated as interchangeable raw measurements.
        Units are harmonised before fusion, and source-specific values and disagreement are kept
        as model features.
        """
    )


def render_admin_sidebar():
    with st.sidebar:
        st.divider()
        st.write("**Location:** Mahikeng, North West")
        st.write("**Training:** NASA POWER + Meteostat")
        st.write("**Training history**")
        st.write(f"From {history.index.min().date()} to {history.index.max().date()}")
        st.write(f"{len(history):,} observations")


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------
current_page = render_topbar()
if is_admin:
    render_admin_sidebar()

if current_page == "Forecast":
    render_forecast_page()
elif current_page == "Advice":
    render_advice_page()
elif current_page == "Data":
    render_data_page()
elif current_page == "Model Results":
    render_model_page()
elif current_page == "About":
    render_about_page()
else:
    render_dashboard()

st.divider()
st.caption("TemoSense · Mahikeng Weather Prediction System")