import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.core.config import get_settings
from app.core.schemas import (
    CityResponse,
    ForecastResponse,
    MethodologyResponse,
    RiskResponse,
    WeatherResponse,
)
from app.data.cities import CITIES, get_city
from app.services.risk_service import RiskService
from app.services.weather import WeatherService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Explainable extreme-heat intelligence for Uzbekistan.",
)
weather_service = WeatherService(settings)
risk_service = RiskService(weather_service)


def _city_or_404(value: str):
    try:
        return get_city(value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unsupported city: {value}") from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": "0.1.0"}


@app.get("/cities", response_model=list[CityResponse])
async def cities() -> list[CityResponse]:
    return [CityResponse.model_validate(city.dict()) for city in CITIES]


@app.get("/weather/{city}", response_model=WeatherResponse)
async def weather(city: str) -> WeatherResponse:
    return await weather_service.current(_city_or_404(city))


@app.get("/risk/{city}", response_model=RiskResponse)
async def risk(city: str) -> RiskResponse:
    return await risk_service.evaluate(_city_or_404(city))


@app.get("/forecast/{city}", response_model=ForecastResponse)
async def forecast(city: str) -> ForecastResponse:
    return await weather_service.forecast(_city_or_404(city))


@app.get("/methodology", response_model=MethodologyResponse)
async def methodology() -> MethodologyResponse:
    return MethodologyResponse(
        version="baseline-v1",
        summary=(
            "A transparent 0–100 environmental heat-hazard index. Each input is clipped to a "
            "documented range and contributes a fixed maximum number of points."
        ),
        formula={
            "air temperature": "0–24 points across 24–43°C",
            "apparent temperature": "0–28 points across 25–50°C",
            "relative humidity": "0–10 points across 30–80%",
            "low wind": "0–8 points, decreasing across 2–22 km/h",
            "solar radiation": "0–8 points across 250–950 W/m²",
            "land-surface temperature": "0–10 points across 30–55°C when available",
            "low NDVI": "0–7 points, decreasing across NDVI 0.1–0.7",
            "low elevation": "0–3 points, decreasing across 0–1,500 m",
            "built-up fraction": "0–2 points across 0.1–0.8",
        },
        thresholds={
            "Safe": "0–19.9",
            "Moderate": "20–39.9",
            "High": "40–59.9",
            "Very High": "60–79.9",
            "Extreme": "80–100",
        },
        limitations=[
            "This is a relative decision-support index, not a clinical prediction.",
            "Fallback geospatial values are static demonstrations and are labeled as such.",
            "Local shade, clothing, workload, health, and indoor conditions are not modeled.",
        ],
    )


@app.get("/model/metrics")
async def model_metrics() -> dict:
    path = Path(settings.metrics_path)
    if not path.exists():
        return {"status": "not_trained", "message": "Run python scripts/train.py to create measured metrics."}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Model metrics artifact is invalid") from exc
