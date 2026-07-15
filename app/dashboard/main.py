import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# Streamlit Cloud executes this nested file directly, so add the repository root
# before importing the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import methodology  # noqa: E402
from app.data.cities import CITIES, City  # noqa: E402
from app.services.risk_service import RiskService  # noqa: E402
from app.services.weather import WeatherService  # noqa: E402

st.set_page_config(page_title="HeatGuard AI Uzbekistan", page_icon="☀️", layout="wide")

RISK_COLORS = {
    "Safe": [22, 163, 74, 220],
    "Moderate": [234, 179, 8, 220],
    "High": [249, 115, 22, 220],
    "Very High": [220, 38, 38, 220],
    "Extreme": [126, 34, 206, 230],
}

TEXT = {
    "en": {
        "subtitle": "Live heat intelligence for safer decisions across Uzbekistan",
        "city": "City",
        "overview": "Overview",
        "details": "City details",
        "forecast": "Forecast",
        "model": "Model performance",
        "method": "Methodology & data",
        "current": "Current conditions",
        "map": "Uzbekistan heat-risk map",
        "fresh": "Data freshness",
        "demo": "Fallback geospatial profile — static demonstration data, not live satellite observations.",
    },
    "uz": {
        "subtitle": "O‘zbekiston bo‘ylab xavfsiz qarorlar uchun issiqlik tahlili",
        "city": "Shahar",
        "overview": "Umumiy ko‘rinish",
        "details": "Shahar tafsilotlari",
        "forecast": "Prognoz",
        "model": "Model natijalari",
        "method": "Metodologiya va ma’lumotlar",
        "current": "Joriy holat",
        "map": "O‘zbekiston issiqlik xavfi xaritasi",
        "fresh": "Ma’lumot yangiligi",
        "demo": "Zaxira geo-profil — statik namoyish ma’lumoti, jonli sun’iy yo‘ldosh kuzatuvi emas.",
    },
}


@st.cache_data(ttl=120, show_spinner=False)
def load_risks() -> list[dict]:
    async def gather() -> list[dict]:
        service = RiskService()
        results = await asyncio.gather(*(service.evaluate(city) for city in CITIES))
        return [item.model_dump(mode="json") for item in results]

    return asyncio.run(gather())


@st.cache_data(ttl=120, show_spinner=False)
def load_forecast(city_slug: str) -> dict:
    city = next(item for item in CITIES if item.slug == city_slug)
    return asyncio.run(WeatherService().forecast(city)).model_dump(mode="json")


@st.cache_data(show_spinner=False)
def load_history(city_slug: str) -> pd.DataFrame:
    path = Path("data/processed/training_data.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["time"])
    return frame.loc[frame["city"] == city_slug, ["time", "risk_score_target"]].tail(365)


def city_label(city: City, language: str) -> str:
    return city.name_uz if language == "uz" else city.name_en


def risk_frame(risks: list[dict]) -> pd.DataFrame:
    rows = []
    for city, risk in zip(CITIES, risks, strict=True):
        rows.append(
            {
                "city": city.name_en,
                "slug": city.slug,
                "latitude": city.latitude,
                "longitude": city.longitude,
                "score": risk["score"],
                "category": risk["category"],
                "temperature": risk["weather"]["temperature_c"],
                "color": RISK_COLORS[risk["category"]],
                "radius": 19000 + risk["score"] * 450,
            }
        )
    return pd.DataFrame(rows)


def render_map(frame: pd.DataFrame) -> None:
    layer = pdk.Layer(
        "ScatterplotLayer",
        frame,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=41.0, longitude=64.5, zoom=4.7),
            layers=[layer],
            tooltip={"html": "<b>{city}</b><br/>{category} · {score}/100<br/>{temperature}°C"},
        ),
        width="stretch",
    )


def render_metrics(risk: dict) -> None:
    weather = risk["weather"]
    columns = st.columns(5)
    columns[0].metric("Risk score", f"{risk['score']:.0f}/100", risk["category"])
    columns[1].metric("Air temperature", f"{weather['temperature_c']:.1f} °C")
    columns[2].metric("Feels like", f"{weather['apparent_temperature_c']:.1f} °C")
    columns[3].metric("Humidity", f"{weather['relative_humidity_pct']:.0f}%")
    columns[4].metric("Wind", f"{weather['wind_speed_kmh']:.1f} km/h")


def render_recommendations(risk: dict) -> None:
    st.subheader("Practical guidance")
    for group in risk["recommendations"]:
        with st.expander(group["audience"], expanded=True):
            for item in group["items"]:
                st.markdown(f"- {item}")


st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#f7faf7 0%,#eef4ef 100%)}
    [data-testid="stMetric"] {background:white;border:1px solid #dce7de;border-radius:14px;padding:14px}
    .hero {padding:1.2rem 0 .5rem}.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:#19764a;font-weight:700}
    .source {padding:.65rem 1rem;border-radius:10px;background:#fff7df;border:1px solid #efd88c}
    </style>
    """,
    unsafe_allow_html=True,
)

language = st.sidebar.segmented_control("Language / Til", options=["en", "uz"], default="en")
t = TEXT[language]
selected_city = st.sidebar.selectbox(t["city"], CITIES, format_func=lambda city: city_label(city, language))
if st.sidebar.button("Refresh live data", width="stretch"):
    load_risks.clear()
    load_forecast.clear()
    st.rerun()
st.sidebar.caption("HeatGuard AI · baseline v1")

st.markdown('<div class="hero"><div class="eyebrow">Climate decision support</div></div>', unsafe_allow_html=True)
st.title("☀️ HeatGuard AI Uzbekistan")
st.caption(t["subtitle"])

with st.spinner("Loading current conditions…"):
    all_risks = load_risks()
selected_risk = next(item for item in all_risks if item["city"] == selected_city.slug)
frame = risk_frame(all_risks)

tabs = st.tabs([t["overview"], t["details"], t["forecast"], t["model"], t["method"]])
with tabs[0]:
    st.subheader(t["map"])
    render_map(frame)
    st.caption("Markers include text labels in the tooltip; color is not the only risk indicator.")
    render_metrics(selected_risk)
    source = selected_risk["weather"]["source"]
    st.markdown(
        f'<div class="source"><b>{t["fresh"]}:</b> {selected_risk["weather"]["fetched_at"]}<br>{source}</div>',
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader(f"{t['current']} · {city_label(selected_city, language)}")
    render_metrics(selected_risk)
    weather = selected_risk["weather"]
    if not weather["is_live"]:
        st.error(
            "Live Open-Meteo data is currently unavailable. The weather values above are an offline "
            "demonstration and are NOT current observations. Use 'Refresh live data' to retry."
        )
    else:
        st.caption(f"Live source: {weather['source']} · observed {weather['observed_at']}")
    geo = selected_risk["geospatial"]
    if geo["is_demo"]:
        st.warning(t["demo"])
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("NDVI", "—" if geo["ndvi"] is None else f"{geo['ndvi']:.2f}")
    g2.metric(
        "Land-surface temp.",
        "—" if geo["land_surface_temperature_c"] is None else f"{geo['land_surface_temperature_c']:.1f} °C",
    )
    g3.metric("Elevation", "—" if geo["elevation_m"] is None else f"{geo['elevation_m']:.0f} m")
    g4.metric("Built-up indicator", "—" if geo["built_up_fraction"] is None else f"{geo['built_up_fraction']:.0%}")
    drivers = pd.DataFrame(
        {"Driver": list(selected_risk["drivers"]), "Points": list(selected_risk["drivers"].values())}
    )
    st.plotly_chart(
        px.bar(drivers, x="Points", y="Driver", orientation="h", title="Risk contribution (percentage points)"),
        width="stretch",
    )
    history = load_history(selected_city.slug)
    if not history.empty:
        st.plotly_chart(
            px.line(
                history,
                x="time",
                y="risk_score_target",
                title="Historical 14:00 risk trend · latest 365 archived days",
                labels={"time": "Date", "risk_score_target": "Baseline risk score"},
                range_y=[0, 100],
            ),
            width="stretch",
        )
    render_recommendations(selected_risk)

with tabs[2]:
    forecast = load_forecast(selected_city.slug)
    forecast_df = pd.DataFrame(forecast["points"])
    forecast_df["time"] = pd.to_datetime(forecast_df["time"])
    st.caption(f"Source: {forecast['source']} · fetched {forecast['fetched_at']}")
    fig = px.line(
        forecast_df,
        x="time",
        y=["temperature_c", "apparent_temperature_c"],
        labels={"value": "Temperature °C", "time": "Local time"},
    )
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(
        px.area(
            forecast_df,
            x="time",
            y="risk_score",
            color_discrete_sequence=["#dd5129"],
            range_y=[0, 100],
            title="Forecast heat-risk score",
        ),
        width="stretch",
    )

with tabs[3]:
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.subheader(f"Selected model: {metrics['selected_model']}")
        st.caption(f"Measured on a held-out time period · {metrics['generated_at']}")
        st.dataframe(pd.DataFrame(metrics["models"]).T, width="stretch")
        if metrics.get("feature_importance"):
            importance = pd.DataFrame(metrics["feature_importance"].items(), columns=["Feature", "Importance"])
            st.plotly_chart(
                px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h"),
                width="stretch",
            )
        st.json(metrics["classification"])
    else:
        st.info("No model report is present yet. Run the documented training command to create measured results.")

with tabs[4]:
    method = asyncio.run(methodology()).model_dump()
    st.subheader("Transparent baseline methodology")
    st.write(method["summary"])
    st.table(pd.DataFrame(method["formula"].items(), columns=["Input", "Contribution"]))
    st.subheader("Risk thresholds")
    st.table(pd.DataFrame(method["thresholds"].items(), columns=["Category", "Score range"]))
    st.subheader("Data sources")
    st.markdown(
        "- **Weather:** Open-Meteo current conditions and hourly forecast (live when reachable).\n"
        "- **Optional satellite:** Google Earth Engine Sentinel-2 NDVI, Landsat 9 LST, and SRTM elevation.\n"
        "- **Fallback:** documented static city profiles, visibly labeled as demonstration data."
    )
    st.subheader("Limitations & disclaimer")
    for limitation in method["limitations"]:
        st.markdown(f"- {limitation}")
    st.warning(selected_risk["disclaimer"])
