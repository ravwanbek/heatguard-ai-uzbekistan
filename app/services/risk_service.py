from app.core.schemas import RiskResponse
from app.data.cities import City
from app.features.risk import DISCLAIMER, calculate_risk, recommendations
from app.services.geospatial import EarthEngineService
from app.services.weather import WeatherService


class RiskService:
    def __init__(self, weather: WeatherService | None = None, geo: EarthEngineService | None = None) -> None:
        self.weather = weather or WeatherService()
        self.geo = geo or EarthEngineService()

    async def evaluate(self, city: City) -> RiskResponse:
        weather = await self.weather.current(city)
        geospatial = self.geo.fetch(city, weather.temperature_c)
        score = calculate_risk(weather, geospatial)
        return RiskResponse(
            city=city.slug,
            score=score.score,
            category=score.category,
            weather=weather,
            geospatial=geospatial,
            drivers=score.drivers,
            recommendations=recommendations(score.category),
            disclaimer=DISCLAIMER,
        )
