"""Compare baseline regressors on a held-out future period and persist measured results."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from app.features.risk import category_for_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/training_data.csv")
MODEL_DIR = Path("models")
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "shortwave_radiation",
    "land_surface_temperature_c",
    "ndvi",
    "elevation_m",
    "built_up_fraction",
    "latitude",
    "longitude",
    "day_of_year_sin",
    "day_of_year_cos",
]


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    day = frame["time"].dt.dayofyear
    frame["day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)
    # The newest full year is never used during fitting or model selection.
    validation_start = frame["time"].max() - pd.DateOffset(years=1)
    train = frame.loc[frame["time"] < validation_start].copy()
    validation = frame.loc[frame["time"] >= validation_start].copy()
    if train.empty or validation.empty:
        raise ValueError("Dataset must span more than one year for a time-based split")
    return train, validation


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    train, validation = prepare(frame)
    x_train, y_train = train[FEATURES], train["risk_score_target"]
    x_val, y_val = validation[FEATURES], validation["risk_score_target"]
    models = {
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "xgboost": xgb.XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=350, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1
        ),
    }
    reports: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict(x_val), 0, 100)
        predictions[name] = prediction
        reports[name] = {
            "mae": round(float(mean_absolute_error(y_val, prediction)), 4),
            "rmse": round(float(mean_squared_error(y_val, prediction) ** 0.5), 4),
            "r2": round(float(r2_score(y_val, prediction)), 4),
        }
    selected = min(reports, key=lambda name: reports[name]["rmse"])
    best = models[selected]
    prediction = predictions[selected]
    true_classes = validation["risk_category_target"].tolist()
    predicted_classes = [category_for_score(float(value)) for value in prediction]
    labels = ["Safe", "Moderate", "High", "Very High", "Extreme"]
    classification = {
        "accuracy": round(float(accuracy_score(true_classes, predicted_classes)), 4),
        "macro_f1": round(
            float(f1_score(true_classes, predicted_classes, labels=labels, average="macro", zero_division=0)), 4
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(true_classes, predicted_classes, labels=labels).tolist(),
    }
    importance_values = getattr(best, "feature_importances_", np.zeros(len(FEATURES)))
    importance = {feature: round(float(value), 5) for feature, value in zip(FEATURES, importance_values, strict=True)}
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "source": "Open-Meteo Historical Weather API; static demo geospatial context",
            "target": "documented baseline-v1 rule score",
            "rows": int(len(frame)),
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "validation_start": validation["time"].min().isoformat(),
            "split": "held-out newest 365 days",
        },
        "models": reports,
        "selected_model": selected,
        "selection_rule": "lowest validation RMSE; LightGBM is not preferred unless it wins",
        "classification": classification,
        "feature_importance": importance,
        "shap": "Not generated in the minimal runtime; tree feature importance is included.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best, "features": FEATURES, "version": "baseline-v1"}, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Selected %s and saved measured report", selected)


if __name__ == "__main__":
    main()
