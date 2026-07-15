from dataclasses import dataclass

from app.core.schemas import GeoFeatures, RecommendationSet, RiskCategory, WeatherResponse

DISCLAIMER = (
    "HeatGuard provides informational decision support only. It is not a medical diagnosis "
    "or a substitute for official weather, workplace-safety, or emergency guidance."
)


def _scale(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low)))


def category_for_score(score: float) -> RiskCategory:
    if score < 20:
        return "Safe"
    if score < 40:
        return "Moderate"
    if score < 60:
        return "High"
    if score < 80:
        return "Very High"
    return "Extreme"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    category: RiskCategory
    drivers: dict[str, float]


def calculate_risk(weather: WeatherResponse, geo: GeoFeatures) -> ScoreResult:
    """Transparent hazard score; component values are percentage-point contributions."""
    components = {
        "air_temperature": 24 * _scale(weather.temperature_c, 24, 43),
        "apparent_temperature": 28 * _scale(weather.apparent_temperature_c, 25, 50),
        "humidity": 10 * _scale(weather.relative_humidity_pct, 30, 80),
        "low_wind": 8 * (1 - _scale(weather.wind_speed_kmh, 2, 22)),
        "solar_radiation": 8 * _scale(weather.shortwave_radiation_wm2 or 0, 250, 950),
        "land_surface_temperature": 10 * _scale(geo.land_surface_temperature_c or 25, 30, 55),
        "low_vegetation": 7 * (1 - _scale(geo.ndvi if geo.ndvi is not None else 0.45, 0.1, 0.7)),
        "low_elevation": 3 * (1 - _scale(geo.elevation_m or 0, 0, 1500)),
        "built_up": 2 * _scale(geo.built_up_fraction or 0, 0.1, 0.8),
    }
    score = round(min(100.0, sum(components.values())), 1)
    return ScoreResult(score, category_for_score(score), {k: round(v, 1) for k, v in components.items()})


def recommendations(category: RiskCategory) -> list[RecommendationSet]:
    level = {"Safe": 0, "Moderate": 1, "High": 2, "Very High": 3, "Extreme": 4}[category]
    construction = ["Carry water and check conditions before each shift."]
    agriculture = ["Prefer early-morning or evening irrigation to reduce evaporation."]
    vulnerable = ["Keep drinking water available and use shade or cooled spaces."]
    if level >= 1:
        construction.append("Schedule heavier tasks for cooler morning hours and take shaded breaks.")
        agriculture.append("Check workers and livestock more often for heat stress.")
        vulnerable.append("Reduce prolonged direct-sun exposure, especially around midday.")
    if level >= 2:
        construction.append("Use a work/rest cycle, buddy checks, and frequent hydration reminders.")
        agriculture.append("Avoid strenuous field work during the hottest hours.")
        vulnerable.append("Children and older adults should remain in a cool place when possible.")
    if level >= 3:
        construction.append("Limit non-essential midday outdoor work and provide active cooling.")
        agriculture.append("Move essential work to dawn or after sunset where safe.")
        vulnerable.append("Arrange regular check-ins; seek urgent help for confusion, fainting, or hot dry skin.")
    if level >= 4:
        construction.append("Postpone heavy outdoor work unless an approved heat-safety plan is active.")
        agriculture.append("Pause non-essential exposed work and prioritize water access for people and animals.")
        vulnerable.append("Stay in an air-conditioned or actively cooled place; follow official alerts.")
    return [
        RecommendationSet(audience="Outdoor & construction workers", items=construction),
        RecommendationSet(audience="Farmers", items=agriculture),
        RecommendationSet(audience="Older people & children", items=vulnerable),
    ]
