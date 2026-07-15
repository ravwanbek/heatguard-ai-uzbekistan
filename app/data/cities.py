from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class City:
    slug: str
    name_en: str
    name_uz: str
    latitude: float
    longitude: float
    elevation_m: float
    ndvi_demo: float
    built_up_demo: float
    lst_offset_demo_c: float

    def dict(self) -> dict[str, str | float]:
        return asdict(self)


CITIES = (
    City("tashkent", "Tashkent", "Toshkent", 41.2995, 69.2401, 455, 0.31, 0.72, 5.1),
    City("samarkand", "Samarkand", "Samarqand", 39.6542, 66.9597, 702, 0.28, 0.51, 5.7),
    City("bukhara", "Bukhara", "Buxoro", 39.7681, 64.4556, 225, 0.16, 0.55, 7.2),
    City("nukus", "Nukus", "Nukus", 42.4619, 59.6166, 76, 0.13, 0.43, 7.6),
    City("andijan", "Andijan", "Andijon", 40.7821, 72.3442, 500, 0.39, 0.47, 4.6),
    City("fergana", "Fergana", "Farg‘ona", 40.3894, 71.7870, 580, 0.42, 0.45, 4.3),
    City("namangan", "Namangan", "Namangan", 40.9983, 71.6726, 476, 0.37, 0.49, 4.8),
    City("qarshi", "Qarshi", "Qarshi", 38.8606, 65.7891, 374, 0.18, 0.46, 6.8),
    City("termez", "Termez", "Termiz", 37.2242, 67.2783, 302, 0.19, 0.42, 7.0),
    City("urgench", "Urgench", "Urganch", 41.5500, 60.6333, 91, 0.22, 0.44, 6.9),
    City("jizzakh", "Jizzakh", "Jizzax", 40.1158, 67.8422, 378, 0.25, 0.39, 5.9),
    City("navoi", "Navoi", "Navoiy", 40.1039, 65.3688, 382, 0.14, 0.48, 7.1),
)

CITY_BY_SLUG = {city.slug: city for city in CITIES}


def get_city(value: str) -> City:
    slug = value.strip().lower().replace(" ", "-")
    if slug not in CITY_BY_SLUG:
        raise KeyError(value)
    return CITY_BY_SLUG[slug]
