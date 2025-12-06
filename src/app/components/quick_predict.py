# src/app/components/quick_predict.py
from __future__ import annotations

from typing import Dict, List, Any

import pandas as pd
import streamlit as st

from src.models.nn_inference import predict_und_2a_from_raw
from src.app.components.metric_card import metric_card

# Variables de negocio que queremos exponer en el formulario (6 columnas)
CAT_FOR_FORM: List[str] = ["mp_categoria", "Tipo_TEJ", "planta_id", "maq_id"]
NUM_FOR_FORM: List[str] = ["Col", "Tal"]


def _pretty_label(col_name: str) -> str:
    """Etiquetas amigables para negocio en el formulario."""
    mapping = {
        "mp_categoria": "Categoría MP",
        "Tipo_TEJ": "Tipo de tejido",
        "planta_id": "Planta",
        "maq_id": "Máquina / Línea",
        "Col": "Cantidad lote (Col)",
        "Tal": "Parámetro técnico (Tal)",
    }
    return mapping.get(col_name, col_name)


def quick_prediction_card(
    X_clean: pd.DataFrame,
    embed_cols: List[str],
    num_cols: List[str],
    model,
    artefacts: Dict[str, Any],
) -> None:
    """
    Renderiza la sección de Quick Prediction – What-if.

    - X_clean: DataFrame ya reorganizado (mismas columnas que usó el modelo).
    - embed_cols: columnas categóricas usadas para embeddings.
    - num_cols: columnas numéricas del modelo.
    - model: modelo Keras ya cargado.
    - artefacts: diccionario con encoders, scaler, etc.
    """

    st.markdown("### Quick Prediction – What-if")

    st.markdown(
        "Ajusta algunos parámetros de entrada para estimar la "
        "**Und_2a_percentage** con el modelo entrenado. "
        "El resto de variables se mantienen en valores típicos del histórico."
    )

    # Filtramos solo las columnas que realmente existen en el modelo
    cat_cols = [c for c in CAT_FOR_FORM if c in embed_cols]
    num_feats = [c for c in NUM_FOR_FORM if c in num_cols]

    if not cat_cols and not num_feats:
        st.warning(
            "No se encontraron columnas válidas para el formulario de predicción. "
            "Revisa CAT_FOR_FORM y NUM_FOR_FORM."
        )
        return

    with st.form("quick_prediction_form"):
        # 2 filas de 3 columnas: 4 categóricas + 2 numéricas
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        raw_input: Dict[str, Any] = {}

        # ======================
        # CATEGÓRICAS
        # ======================
        # mp_categoria
        if len(cat_cols) >= 1:
            col_name = cat_cols[0]
            options = (
                X_clean[col_name]
                .dropna()
                .unique()
                .tolist()
            )
            with r1c1:
                raw_input[col_name] = st.selectbox(
                    _pretty_label(col_name),
                    options,
                    format_func=str,  # UI en str pero mantiene el tipo original
                )

        # Tipo_TEJ
        if len(cat_cols) >= 2:
            col_name = cat_cols[1]
            options = (
                X_clean[col_name]
                .dropna()
                .unique()
                .tolist()
            )
            with r1c2:
                raw_input[col_name] = st.selectbox(
                    _pretty_label(col_name),
                    options,
                    format_func=str,
                )

        # planta_id
        if len(cat_cols) >= 3:
            col_name = cat_cols[2]
            options = (
                X_clean[col_name]
                .dropna()
                .unique()
                .tolist()
            )
            with r1c3:
                raw_input[col_name] = st.selectbox(
                    _pretty_label(col_name),
                    options,
                    format_func=str,
                )

        # maq_id
        if len(cat_cols) >= 4:
            col_name = cat_cols[3]
            options = (
                X_clean[col_name]
                .dropna()
                .unique()
                .tolist()
            )
            with r2c1:
                raw_input[col_name] = st.selectbox(
                    _pretty_label(col_name),
                    options,
                    format_func=str,
                )

        # ======================
        # NUMÉRICAS
        # ======================
        # Col
        if len(num_feats) >= 1:
            col_name = num_feats[0]
            default_val = float(X_clean[col_name].median())
            with r2c2:
                raw_input[col_name] = st.number_input(
                    _pretty_label(col_name),
                    value=default_val,
                )

        # Tal
        if len(num_feats) >= 2:
            col_name = num_feats[1]
            default_val = float(X_clean[col_name].median())
            with r2c3:
                raw_input[col_name] = st.number_input(
                    _pretty_label(col_name),
                    value=default_val,
                )

        # ======================
        # Construcción del row completo
        # ======================
        base_row = X_clean.iloc[0].to_dict()
        for k, v in raw_input.items():
            base_row[k] = v

        submitted = st.form_submit_button("Calcular predicción")

    if submitted:
        y_hat = predict_und_2a_from_raw(base_row, model, artefacts)

        # Texto adicional en negrita
        st.markdown(
            f"**Predicción estimada de Und_2a_percentage:** `{y_hat:.4f}`"
        )

        # KPI tipo tarjeta (usa el estilo de metric_card)
        metric_card(
            title="Predicted Und_2a_percentage",
            value=f"{y_hat:.4f}",
            subtitle="Zero-Inflated NN · quick what-if",
        )
