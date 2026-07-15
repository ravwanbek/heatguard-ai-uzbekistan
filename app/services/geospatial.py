import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.schemas import GeoFeatures
from app.data.cities import City

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/processed/gee_cache")


def fallback_features(city: City, air_temperature_c: float | None) -> GeoFeatures:
    """Documented static city context; LST is a demo estimate, not satellite data."""
    lst = None if air_temperature_c is None else round(air_temperature_c + city.lst_offset_demo_c, 1)
    return GeoFeatures(
        ndvi=city.ndvi_demo,
        land_surface_temperature_c=lst,
        elevation_m=city.elevation_m,
        built_up_fraction=city.built_up_demo,
        source="Static demonstration geospatial profile; LST is estimated, not a satellite observation",
        is_demo=True,
    )


class EarthEngineService:
    """Optional Earth Engine adapter with credential-aware caching and graceful fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def configured(self) -> bool:
        return bool(self.settings.gee_project)

    def fetch(self, city: City, air_temperature_c: float | None = None) -> GeoFeatures:
        cache_file = CACHE_DIR / f"{city.slug}.json"
        if cache_file.exists():
            try:
                return GeoFeatures.model_validate(json.loads(cache_file.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                logger.warning("Ignoring invalid Earth Engine cache for %s", city.slug)
        if not self.configured:
            return fallback_features(city, air_temperature_c)
        try:
            import ee

            self._initialize(ee)
            point = ee.Geometry.Point([city.longitude, city.latitude])
            area = point.buffer(5000)
            end = datetime.now(UTC).date().isoformat()
            start = (datetime.now(UTC).date().replace(day=1)).isoformat()
            sentinel = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(area)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .median()
            )
            ndvi = sentinel.normalizedDifference(["B8", "B4"]).rename("ndvi")
            landsat = (
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterBounds(area)
                .filterDate(start, end)
                .sort("CLOUD_COVER")
                .first()
            )
            lst = ee.Image(landsat).select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
            elevation = ee.Image("USGS/SRTMGL1_003")
            scale = 100
            reducer = ee.Reducer.mean()
            result = GeoFeatures(
                ndvi=ndvi.reduceRegion(reducer, area, scale, bestEffort=True).get("ndvi").getInfo(),
                land_surface_temperature_c=lst.reduceRegion(reducer, area, scale, bestEffort=True)
                .get("ST_B10")
                .getInfo(),
                elevation_m=elevation.reduceRegion(reducer, area, 30, bestEffort=True).get("elevation").getInfo(),
                built_up_fraction=city.built_up_demo,
                source="Google Earth Engine: Sentinel-2, Landsat 9 Collection 2, SRTM",
                is_demo=False,
                observed_at=datetime.now(UTC),
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result
        except Exception as exc:  # Earth Engine raises several optional-package exceptions.
            logger.warning("Earth Engine unavailable for %s: %s", city.slug, exc)
            return fallback_features(city, air_temperature_c)

    def _initialize(self, ee: object) -> None:
        if self.settings.gee_service_account and self.settings.gee_private_key_file:
            credentials = ee.ServiceAccountCredentials(  # type: ignore[attr-defined]
                self.settings.gee_service_account, self.settings.gee_private_key_file
            )
            ee.Initialize(credentials, project=self.settings.gee_project)  # type: ignore[attr-defined]
        else:
            ee.Initialize(project=self.settings.gee_project)  # type: ignore[attr-defined]
