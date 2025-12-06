from typing import Dict, Any
import pandas as pd
import joblib

from src.config.settings import MODELS_DIR

_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODELS_DIR / "modelo_xgb_und2a.pkl")
    return _pipeline

def predict_one(input_dict: Dict[str, Any]) -> float:
    pipeline = _load_pipeline()
    df = pd.DataFrame([input_dict])
    pred = pipeline.predict(df)[0]
    return float(pred)

def predict_batch(df: pd.DataFrame) -> pd.Series:
    pipeline = _load_pipeline()
    preds = pipeline.predict(df)
    return pd.Series(preds, index=df.index, name="pred_und2a_percentage")
