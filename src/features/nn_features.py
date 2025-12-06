# src/features/nn_features.py
"""
Reorganización de features para el modelo NN:
- elimina columnas de ruido y leakage
- separa columnas para embeddings vs numéricas escaladas.
"""

import pandas as pd
from typing import List, Tuple

from src.config.nn_config import (
    NOISE_COLS,
    USER_NUMERICAL,
    USER_CATEGORICAL,
    LEAKAGE_COLS,
)


def reorganize_features_final(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Reorganiza el dataframe en:
    - df_clean: sin ruido ni leakage
    - embed_cols: columnas que irán a embeddings (IDs + categorías)
    - num_cols: columnas numéricas puras

    Decisión de diseño:
    - IDs como embeddings reduce la dimensionalidad respecto a One-Hot y captura similitud entre categorías (Goodfellow et al., 2016).
    """
    df = df.copy()

    # 1. Eliminar leakage y ruido
    cols_to_drop = [c for c in (LEAKAGE_COLS + NOISE_COLS) if c in df.columns]
    if cols_to_drop:
        print(f"Eliminando ruido/leakage: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    potential_ids = ["planta_id", "seccion_id", "maq_id", "producto_id", "estilo_id", "mp_id"]

    final_embeddings = list(USER_CATEGORICAL)
    final_numerics: List[str] = []

    for col in USER_NUMERICAL:
        if col in cols_to_drop or col not in df.columns:
            continue

        if col in potential_ids:
            final_embeddings.append(col)
        else:
            final_numerics.append(col)

    # Filtrar por columnas realmente presentes
    final_embeddings = [c for c in final_embeddings if c in df.columns]
    final_numerics = [c for c in final_numerics if c in df.columns]

    # Quitar duplicados por seguridad
    final_embeddings = list(set(final_embeddings))
    final_numerics = list(set([c for c in final_numerics if c not in final_embeddings]))

    print(f"Variables finales: {len(final_embeddings) + len(final_numerics)}")

    return df, final_embeddings, final_numerics
