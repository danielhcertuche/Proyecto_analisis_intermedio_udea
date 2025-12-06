# src/models/nn_preprocessing.py
"""
Preprocesamiento para el modelo NN:
- LabelEncoder por columna categórica (para embeddings)
- StandardScaler para numéricas.
"""

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    embed_cols: List[str],
    num_cols: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, LabelEncoder], int, StandardScaler | None]:
    """
    Construye diccionarios de inputs para Keras:
    - 'in_<col>' para columnas de embeddings
    - 'in_numerics' para bloque numérico escalado

    Patrón:
    - Ajustamos encoders + scaler solo en train (Hastie et al., 2009).
    - Definimos explícitamente un token <UNK> para categorías nuevas en test.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    input_train: Dict[str, np.ndarray] = {}
    input_test: Dict[str, np.ndarray] = {}
    encoders: Dict[str, LabelEncoder] = {}

    # A. Embeddings
    for col in embed_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        le = LabelEncoder()
        train_vals = list(X_train[col].unique())
        train_vals.append("<UNK>")
        le.fit(train_vals)
        encoders[col] = le

        input_train[f"in_{col}"] = le.transform(X_train[col])

        unk_idx = len(le.classes_) - 1
        input_test[f"in_{col}"] = np.array(
            [le.transform([val])[0] if val in le.classes_ else unk_idx for val in X_test[col]]
        )

    # B. Numéricas
    scaler: StandardScaler | None = None
    n_numeric_features: int = 0

    if num_cols:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_test_num = scaler.transform(X_test[num_cols])

        input_train["in_numerics"] = X_train_num
        input_test["in_numerics"] = X_test_num
        n_numeric_features = X_train_num.shape[1]

    return input_train, input_test, encoders, n_numeric_features, scaler
