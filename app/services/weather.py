import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx

from app.core.config import Settings, get_settings
from app.core.schemas import ForecastPoint, ForecastResponse, WeatherResponse
from app.data.cities import City
from app.features.risk import calculate_risk, category_for_score
from app.services.geospatial import fallback_features

logger = logging.getLogger(__name__)

CURRENT_VARS = "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,shortwave_radiation"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"


class WeatherService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[str, tuple[datetime, WeatherResponse]] = {}

    async def _request_json(self, params: dict[str, str | float | int]) -> dict:
        """Request Open-Meteo with a retry for transient cloud/network failures."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.request_timeout_seconds,
                    follow_redirects=True,
                    headers={"User-Agent": "HeatGuard-AI-Uzbekistan/0.1"},
                ) as client:
                    response = await client.get(self.settings.open_meteo_base_url, params=params)
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.75 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Open-Meteo request failed without an error")

    async def current(self, city: City) -> WeatherResponse:
        cached = self._cache.get(city.slug)
        now = datetime.now(UTC)
        if cached and (now - cached[0]).total_seconds() < self.settings.cache_ttl_seconds:
            return cached[1]
        try:
            payload = await self._request_json(
                {
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "current": CURRENT_VARS,
                    "timezone": "Asia/Tashkent",
                }
            )
            current = payload["current"]
            result = WeatherResponse(
                city=city.slug,
                observed_at=current["time"],
                fetched_at=now,
                temperature_c=current["temperature_2m"],
                relative_humidity_pct=current["relative_humidity_2m"],
                apparent_temperature_c=current["apparent_temperature"],
                wind_speed_kmh=current["wind_speed_10m"],
                shortwave_radiation_wm2=current.get("shortwave_radiation"),
                source="Open-Meteo live forecast API",
                is_live=True,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Weather request failed for %s: %s", city.slug, exc)
            result = demo_weather(city, now)
        # Do not lock a transient failure into the service cache.
        if result.is_live:
            self._cache[city.slug] = (now, result)
        return result

    async def current_all(self, cities: tuple[City, ...]) -> list[WeatherResponse]:
        return list(await asyncio.gather(*(self.current(city) for city in cities)))

    async def forecast(self, city: City, hours: int = 48) -> ForecastResponse:
        now = datetime.now(UTC)
        try:
            payload = await self._request_json(
                {
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "hourly": HOURLY_VARS,
                    "forecast_days": 3,
                    "timezone": "Asia/Tashkent",
                }
            )
            hourly = payload["hourly"]
            points = []
            geo = fallback_features(city, None)
            for i, time in enumerate(hourly["time"][:hours]):
                weather = WeatherResponse(
                    city=city.slug,
                    observed_at=time,
                    fetched_at=now,
                    temperature_c=hourly["temperature_2m"][i],
                    relative_humidity_pct=hourly["relative_humidity_2m"][i],
                    apparent_temperature_c=hourly["apparent_temperature"][i],
                    wind_speed_kmh=hourly["wind_speed_10m"][i],
                    source="Open-Meteo forecast API",
                    is_live=True,
                )
                risk = calculate_risk(weather, geo)
                points.append(
                    ForecastPoint(
                        time=time,
                        temperature_c=weather.temperature_c,
                        apparent_temperature_c=weather.apparent_temperature_c,
                        relative_humidity_pct=weather.relative_humidity_pct,
                        wind_speed_kmh=weather.wind_speed_kmh,
                        risk_score=risk.score,
                        risk_category=risk.category,
                    )
                )
            return ForecastResponse(city=city.slug, source="Open-Meteo forecast API", fetched_at=now, points=points)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Forecast request failed for %s: %s", city.slug, exc)
            return demo_forecast(city, now, hours)


def demo_weather(city: City, now: datetime | None = None) -> WeatherResponse:
    """Deterministic offline demonstration, never represented as an observation."""
    now = now or datetime.now(UTC)
    city_index = sum(ord(char) for char in city.slug) % 9
    temperature = 28.0 + city_index * 0.8
    humidity = 30.0 + (city_index * 4) % 27
    apparent = temperature + max(0, (humidity - 35) * 0.06)
    return WeatherResponse(
        city=city.slug,
        observed_at=now,
        fetched_at=now,
        temperature_c=round(temperature, 1),
        relative_humidity_pct=round(humidity, 1),
        apparent_temperature_c=round(apparent, 1),
        wind_speed_kmh=round(7 + city_index * 0.7, 1),
        shortwave_radiation_wm2=620.0,
        source="Offline demonstration weather (not a live observation)",
        is_live=False,
    )


def demo_forecast(city: City, now: datetime, hours: int) -> ForecastResponse:
    base = demo_weather(city, now)
    geo = fallback_features(city, base.temperature_c)
    points: list[ForecastPoint] = []
    for hour in range(hours):
        local_hour = (now.hour + 5 + hour) % 24
        diurnal = max(-5.0, 7.0 - abs(local_hour - 15) * 1.15)
        temp = base.temperature_c + diurnal - 2
        apparent = temp + max(0, (base.relative_humidity_pct - 35) * 0.06)
        weather = base.model_copy(
            update={
                "observed_at": now + timedelta(hours=hour),
                "temperature_c": temp,
                "apparent_temperature_c": apparent,
            }
        )
        risk = calculate_risk(weather, geo)
        points.append(
            ForecastPoint(
                time=weather.observed_at,
                temperature_c=round(temp, 1),
                apparent_temperature_c=round(apparent, 1),
                relative_humidity_pct=base.relative_humidity_pct,
                wind_speed_kmh=base.wind_speed_kmh,
                risk_score=risk.score,
                risk_category=category_for_score(risk.score),
            )
        )
    return ForecastResponse(city=city.slug, source="Offline demonstration forecast", fetched_at=now, points=points)
