import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config.settings import MODELS_DIR, REPORTS_DIR
from src.data.load_data import load_clean_dataset
from src.features.build_features import (
    temporal_train_test_split,
    remove_target_outliers,
    build_preprocessor,
)
from src.config.model_config import get_xgb_model

def train_xgb_pipeline() -> dict:
    df = load_clean_dataset()

    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    X_train, y_train = remove_target_outliers(X_train, y_train)

    preprocessor = build_preprocessor(X_train)
    model = get_xgb_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path   = MODELS_DIR / "modelo_xgb_und2a.pkl"
    metrics_path = REPORTS_DIR / "metrics_xgb_und2a.json"

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }
