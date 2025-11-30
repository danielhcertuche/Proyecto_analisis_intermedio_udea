"""
preprocessing.py
Módulo con funciones de preprocesamiento de datos.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# 1. Limpieza básica
# ---------------------------------------------------------------------



def parse_dates(df: pd.DataFrame, date_col: str = "anio_mes") -> pd.DataFrame:
    """Convierte columna de fecha a datetime."""
    df_out = df.copy()
    if date_col in df_out.columns:
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    return df_out


def fill_missing_units(
    df: pd.DataFrame,
    col_1a: str = "Und_1a",
    col_2a: str = "Und_2a",
) -> pd.DataFrame:
    """Rellena nulos en columnas de unidades con 0."""
    df_out = df.copy()
    if col_1a in df_out.columns:
        df_out[col_1a] = df_out[col_1a].fillna(0)
    if col_2a in df_out.columns:
        df_out[col_2a] = df_out[col_2a].fillna(0)
    return df_out


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza básica."""
    df_clean = parse_dates(df)
    df_clean = fill_missing_units(df_clean)
    return df_clean


# ---------------------------------------------------------------------
# 2. Utilidades para inspección rápida (sin plots)
# ---------------------------------------------------------------------



def missing_percentage(df: pd.DataFrame) -> pd.Series:
    """Porcentaje de nulos por columna (>0)."""
    miss_pct = df.isnull().sum() / len(df) * 100
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)
    return miss_pct


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Devuelve listas de columnas numéricas y categóricas según dtype."""
    numeric_cols = df.select_dtypes(include=["number", "datetime64[ns]"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------
# 3. Constructor de preprocessor de scikit-learn
# ---------------------------------------------------------------------



def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """ColumnTransformer: num -> StandardScaler, cat -> OneHot."""
    num_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor