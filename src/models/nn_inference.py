# src/models/nn_inference.py
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import joblib
import tensorflow as tf

from src.config.settings import MODELS_DIR
from src.config.nn_config import NN_MODEL_SUBDIR, NN_PIPELINE_PKL
import os
import pathlib

# 🔧 Parche cross-platform: permitir cargar objetos WindowsPath en Linux
if os.name != "nt" and hasattr(pathlib, "WindowsPath"):
    pathlib.WindowsPath = pathlib.PosixPath

def load_nn_zero_inflated_bundle() -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Carga el modelo Keras y los artefactos de preprocesamiento
    (encoders, scaler, listas de columnas) desde el pkl.

    Devuelve:
        model    : modelo Keras ya cargado
        artefacts: dict con keys
            - "keras_model_path"
            - "encoders"
            - "scaler"
            - "embed_cols"
            - "num_cols"
    """
    pipe_path = MODELS_DIR / NN_MODEL_SUBDIR / NN_PIPELINE_PKL
    artefacts: Dict[str, Any] = joblib.load(pipe_path)

    keras_path = artefacts["keras_model_path"]
    model = tf.keras.models.load_model(keras_path)

    return model, artefacts


def build_inputs_from_raw(
    raw_features: Dict[str, Any],
    artefacts: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Construye el diccionario de tensores que espera el modelo Keras
    a partir de un diccionario de valores "crudos" (sin codificar).

    raw_features debe contener al menos todas las columnas:
        artefacts["embed_cols"] + artefacts["num_cols"]

    IMPORTANTE: los nombres de las llaves deben coincidir con los usados
    durante el entrenamiento, por ejemplo:
        'in_mp_categoria', 'in_mp_id', ..., 'in_C', 'in_numerics'
    """
    encoders = artefacts["encoders"]
    scaler = artefacts["scaler"]
    embed_cols = artefacts["embed_cols"]
    num_cols = artefacts["num_cols"]

    inputs: Dict[str, np.ndarray] = {}

    # ---- categóricas -> enteros para embeddings ----
    for col in embed_cols:
        val = raw_features[col]
        encoder = encoders[col]
        classes = encoder.classes_

        # Intentamos castear al mismo tipo que las clases del encoder
        try:
            val_cast = classes.dtype.type(val)
        except Exception:
            val_cast = val

        encoded = encoder.transform([val_cast])  # -> shape (1,)
        inputs[f"in_{col}"] = encoded.astype("int32")

    # ---- numéricas -> vector escalado ----
    num_vals = np.array([[raw_features[col] for col in num_cols]], dtype="float32")
    num_scaled = scaler.transform(num_vals)
    inputs["in_numerics"] = num_scaled.astype("float32")

    return inputs


def predict_und_2a_from_raw(
    raw_features: Dict[str, Any],
    model: tf.keras.Model,
    artefacts: Dict[str, Any],
) -> float:
    """
    Hace una predicción puntual de Und_2a_percentage a partir de un
    diccionario de features "crudos".
    """
    model_inputs = build_inputs_from_raw(raw_features, artefacts)
    pred = model.predict(model_inputs, verbose=0)
    return float(pred.reshape(-1)[0])
