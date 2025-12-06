# src/models/train_nn_zero.py
"""
Pipeline de entrenamiento para la NN zero-inflated:
- carga dataset limpio
- organiza features
- preprocesa (encoders + scaler)
- entrena y guarda artefactos en:
    * MODELS_DIR/nn_zero_inflated  (modelo + pipeline)
    * reports/                     (historia + métricas + hyperparams)
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras import callbacks

from src.config.settings import (
    TARGET_COL,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.config.nn_config import (
    NN_MODEL_SUBDIR,
    NN_KERAS_NAME,
    NN_PIPELINE_PKL,
)
from src.data.load_data import load_clean_dataset
from src.features.nn_features import reorganize_features_final
from src.models.nn_preprocessing import preprocess_data
from src.models.nn_zero_inflated import build_dynamic_model_tuned


def train_nn_zero_inflated() -> Dict[str, Any]:
    # --- 1. Datos base ---
    df = load_clean_dataset()
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    # --- 2. Selección y reordenamiento de features ---
    X_clean, embed_cols, final_num_cols = reorganize_features_final(X)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # --- 3. Preprocesamiento (encoders + scaler) ---
    train_inputs, test_inputs, encoders, n_nums, scaler = preprocess_data(
        X_train_raw,
        X_test_raw,
        embed_cols,
        final_num_cols,
    )

    # --- 4. Modelo NN zero-inflated ---
    model = build_dynamic_model_tuned(
        embed_cols=embed_cols,
        encoders=encoders,
        n_numeric_features=n_nums,
        learning_rate=3e-4,
    )

    cb = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=4),
    ]

    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(test_inputs, y_test),
        epochs=50,
        batch_size=32,
        callbacks=cb,
        verbose=1,
    )

    # --- 5. Evaluación hold-out ---
    preds = model.predict(test_inputs).reshape(-1)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"R2   : {r2:.4f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")

    # ==============================
    # 6. Guardado de artefactos
    # ==============================

    # 6.1. Modelo + pipeline (para inferencia)
    model_dir = MODELS_DIR / NN_MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    keras_path = model_dir / NN_KERAS_NAME
    pipe_path = model_dir / NN_PIPELINE_PKL

    model.save(keras_path)
    print(f"Modelo Keras guardado en: {keras_path}")

    pipeline_artefactos = {
        "keras_model_path": keras_path,
        "encoders": encoders,
        "scaler": scaler,
        "embed_cols": embed_cols,
        "num_cols": final_num_cols,
    }
    joblib.dump(pipeline_artefactos, pipe_path)
    print(f"Pipeline completo guardado en: {pipe_path}")

    # 6.2. Artefactos para dashboard / Streamlit
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # historia de entrenamiento
    hist_df = pd.DataFrame(history.history)
    hist_path = REPORTS_DIR / "history_nn_zero.csv"
    hist_df.to_csv(hist_path, index_label="epoch")

    # métricas agregadas
    metrics = {"model_name": "nn_zero_inflated", "r2": float(r2), "mse": float(mse), "rmse": float(rmse)}
    metrics_path = REPORTS_DIR / "metrics_nn_zero.json"
    metrics_path.write_text(
        pd.io.json.dumps(metrics, indent=2),  # usa el json de pandas para evitar imports extra
        encoding="utf-8",
    )

    # hyperparams mínimos (para la tarjeta de configuración)
    hyperparams = {
        "learning_rate": 3e-4,
        "epochs": 50,
        "batch_size": 32,
        "patience_es": 8,
        "patience_lr": 4,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
    }
    hyper_path = REPORTS_DIR / "hyperparams_nn_zero.json"
    hyper_path.write_text(
        pd.io.json.dumps(hyperparams, indent=2),
        encoding="utf-8",
    )

    # tabla de comparación de modelos (append si ya existe)
    comp_path = REPORTS_DIR / "model_comparison.csv"
    row = {"model": "nn_zero_inflated", "R2": r2, "RMSE": rmse}
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = pd.concat([comp_df, pd.DataFrame([row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([row])
    comp_df.to_csv(comp_path, index=False)

    return {
        "history": history.history,
        "metrics": metrics,
        "hyperparams": hyperparams,
        "model_path": keras_path,
        "pipeline_path": pipe_path,
        "history_path": hist_path,
        "metrics_path": metrics_path,
        "hyperparams_path": hyper_path,
        "model_comparison_path": comp_path,
    }


if __name__ == "__main__":
    _ = train_nn_zero_inflated()
