"""Download historical Open-Meteo observations and create leakage-safe daily samples."""

import argparse
import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from app.core.schemas import WeatherResponse
from app.data.cities import CITIES
from app.features.risk import calculate_risk
from app.services.geospatial import fallback_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_city_year(city, year: int) -> pd.DataFrame:
    end = date(year, 12, 31)
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": f"{year}-01-01",
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,shortwave_radiation",
        "timezone": "Asia/Tashkent",
    }
    response = httpx.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90)
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    # Local 14:00 is representative of workday heat and yields one independent row/day.
    frame = frame.loc[frame["time"].dt.hour == 14].copy()
    frame["city"] = city.slug
    frame["latitude"] = city.latitude
    frame["longitude"] = city.longitude
    frame["elevation_m"] = city.elevation_m
    frame["ndvi"] = city.ndvi_demo
    frame["built_up_fraction"] = city.built_up_demo
    frame["land_surface_temperature_c"] = frame["temperature_2m"] + city.lst_offset_demo_c
    return frame


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    scores: list[float] = []
    categories: list[str] = []
    city_map = {city.slug: city for city in CITIES}
    for row in frame.itertuples():
        city = city_map[row.city]
        weather = WeatherResponse(
            city=city.slug,
            observed_at=row.time,
            fetched_at=row.time,
            temperature_c=row.temperature_2m,
            relative_humidity_pct=row.relative_humidity_2m,
            apparent_temperature_c=row.apparent_temperature,
            wind_speed_kmh=row.wind_speed_10m,
            shortwave_radiation_wm2=row.shortwave_radiation,
            source="Open-Meteo Historical Weather API",
            is_live=False,
        )
        geo = fallback_features(city, row.temperature_2m)
        result = calculate_risk(weather, geo)
        scores.append(result.score)
        categories.append(result.category)
    result = frame.copy()
    result["risk_score_target"] = scores
    result["risk_category_target"] = categories
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=Path("data/processed/training_data.csv"))
    args = parser.parse_args()
    frames = []
    for city in CITIES:
        for year in range(args.start_year, args.end_year + 1):
            logger.info("Downloading %s %s", city.slug, year)
            frames.append(fetch_city_year(city, year))
    dataset = add_targets(pd.concat(frames, ignore_index=True).dropna())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    logger.info("Saved %s measured historical rows to %s", len(dataset), args.output)


if __name__ == "__main__":
    main()
