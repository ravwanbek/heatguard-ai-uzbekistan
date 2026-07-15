from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskCategory = Literal["Safe", "Moderate", "High", "Very High", "Extreme"]


class CityResponse(BaseModel):
    slug: str
    name_en: str
    name_uz: str
    latitude: float
    longitude: float
    elevation_m: float


class WeatherResponse(BaseModel):
    city: str
    observed_at: datetime
    fetched_at: datetime
    temperature_c: float
    relative_humidity_pct: float
    apparent_temperature_c: float
    wind_speed_kmh: float
    shortwave_radiation_wm2: float | None = None
    source: str
    is_live: bool


class GeoFeatures(BaseModel):
    ndvi: float | None = None
    land_surface_temperature_c: float | None = None
    elevation_m: float | None = None
    built_up_fraction: float | None = None
    source: str
    is_demo: bool
    observed_at: datetime | None = None


class RecommendationSet(BaseModel):
    audience: str
    items: list[str]


class RiskResponse(BaseModel):
    city: str
    score: float = Field(ge=0, le=100)
    category: RiskCategory
    weather: WeatherResponse
    geospatial: GeoFeatures
    drivers: dict[str, float]
    recommendations: list[RecommendationSet]
    methodology_version: str = "baseline-v1"
    disclaimer: str


class ForecastPoint(BaseModel):
    time: datetime
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    risk_score: float = Field(ge=0, le=100)
    risk_category: RiskCategory


class ForecastResponse(BaseModel):
    city: str
    source: str
    fetched_at: datetime
    points: list[ForecastPoint]


class MethodologyResponse(BaseModel):
    version: str
    summary: str
    formula: dict[str, str]
    thresholds: dict[str, str]
    limitations: list[str]
