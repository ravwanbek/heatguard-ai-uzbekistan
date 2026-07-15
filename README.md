# HeatGuard AI Uzbekistan

A working first version of a browser-based climate intelligence platform for 12 Uzbek cities. It combines live Open-Meteo conditions, optional Google Earth Engine features, an explicitly labeled offline/geospatial fallback, explainable risk scoring, practical guidance, a FastAPI API, and a Streamlit dashboard.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env  # Windows; use: cp .env.example .env on macOS/Linux
uvicorn app.api.main:app --reload
```

In a second terminal:

```bash
streamlit run app/dashboard/main.py
```

Open <http://localhost:8501>. API docs are at <http://localhost:8000/docs>. The dashboard deliberately uses the same Python services directly, so it remains usable even if the API process is not running.

## Docker

```bash
copy .env.example .env
docker compose up --build
```

Dashboard: <http://localhost:8501>; API: <http://localhost:8000>.

## Data behavior

- Weather is requested from the free Open-Meteo API and cached for 15 minutes. Network failure activates deterministic **offline demonstration weather**, prominently labeled as non-live.
- Without Earth Engine, NDVI, elevation, built-up fraction, and an air-temperature-based LST estimate come from documented static demonstration profiles. These values are never described as live satellite observations.
- With Earth Engine, the optional adapter requests a 5 km city buffer using Sentinel-2 SR NDVI, Landsat 9 Collection 2 LST, and SRTM elevation, then caches the result.

## Google Earth Engine (optional)

1. Enable Earth Engine for a Google Cloud project and install the optional package: `pip install -e ".[earth-engine]"`.
2. For interactive credentials, run `earthengine authenticate` and set `HEATGUARD_GEE_PROJECT` in `.env`.
3. For a service account, also set `HEATGUARD_GEE_SERVICE_ACCOUNT` and `HEATGUARD_GEE_PRIVATE_KEY_FILE` to the JSON key path.
4. Restart both services. If authentication, imagery, or credentials fail, the application logs the failure and safely returns labeled fallback data.

## Historical dataset and ML

```bash
python scripts/build_dataset.py --start-year 2022 --end-year 2025
python scripts/train.py
```

The first command downloads one local 14:00 observation per city/day from Open-Meteo. The second compares Random Forest, XGBoost, and LightGBM using the newest 365 days as a time holdout. It records MAE, RMSE, R², accuracy, macro F1, a confusion matrix, and feature importance. The lowest holdout RMSE wins; LightGBM is chosen only if supported by that evaluation. Artifacts are written to `models/best_model.joblib` and `models/metrics.json`.

The training label is the documented `baseline-v1` rule score because no reliable health-outcome labels are bundled. Reported measurements therefore quantify approximation of that target, not medical accuracy. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quality checks

```bash
ruff format --check .
ruff check .
pytest
```

## API

- `GET /health`
- `GET /cities`
- `GET /weather/{city}`
- `GET /risk/{city}`
- `GET /forecast/{city}`
- `GET /methodology`
- `GET /model/metrics`

Supported slugs: `tashkent`, `samarkand`, `bukhara`, `nukus`, `andijan`, `fergana`, `namangan`, `qarshi`, `termez`, `urgench`, `jizzakh`, `navoi`.

## Limitations and disclaimer

The score is a relative environmental decision-support index. It does not model personal health, exertion, clothing, shade, building conditions, or every neighborhood microclimate. It is not medical diagnosis and does not replace official weather alerts, workplace procedures, or emergency guidance.
