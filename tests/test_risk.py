from datetime import UTC, datetime

from app.core.schemas import GeoFeatures, WeatherResponse
from app.features.risk import calculate_risk, category_for_score


def weather(temp: float, feels: float, humidity: float = 40) -> WeatherResponse:
    now = datetime.now(UTC)
    return WeatherResponse(
        city="tashkent",
        observed_at=now,
        fetched_at=now,
        temperature_c=temp,
        relative_humidity_pct=humidity,
        apparent_temperature_c=feels,
        wind_speed_kmh=10,
        shortwave_radiation_wm2=600,
        source="test",
        is_live=False,
    )


def test_risk_is_bounded_and_increases_with_heat() -> None:
    geo = GeoFeatures(ndvi=0.3, elevation_m=400, source="test", is_demo=True)
    cool = calculate_risk(weather(24, 24), geo)
    hot = calculate_risk(weather(44, 51, 65), geo)
    assert 0 <= cool.score < hot.score <= 100


def test_thresholds() -> None:
    assert [category_for_score(value) for value in (0, 20, 40, 60, 80)] == [
        "Safe",
        "Moderate",
        "High",
        "Very High",
        "Extreme",
    ]
