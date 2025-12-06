# src/models/importance.py
"""
Permutation Importance para modelos Keras con inputs dict().
"""

from typing import Dict
import numpy as np
import tensorflow as tf


def calculate_permutation_importance(
    model: tf.keras.Model,
    X_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    metric_index: int,
    sample_size: int = 5000,
) -> Dict[str, float]:
    """
    Calcula Importancia por Permutación basada en el aumento del error.

    Basado en la idea de Breiman (2001) para Random Forests,
    adaptado a modelos arbitrarios: romper la relación feature–target
    y medir el aumento en la métrica de interés.
    """
    n = len(y_true)
    idx = np.random.choice(n, min(sample_size, n), replace=False)

    inputs_sample = {k: v[idx].copy() for k, v in X_dict.items()}
    y_sample = y_true[idx]

    baseline = model.evaluate(inputs_sample, y_sample, verbose=0)[metric_index]

    importances: Dict[str, float] = {}

    for key in inputs_sample.keys():
        backup = inputs_sample[key].copy()

        np.random.shuffle(inputs_sample[key])
        score = model.evaluate(inputs_sample, y_sample, verbose=0)[metric_index]

        importances[key] = score - baseline
        inputs_sample[key] = backup

    return importances
