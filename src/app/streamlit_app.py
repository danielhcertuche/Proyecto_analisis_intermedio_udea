# src/app/streamlit_app.py
"""
ML Training Report Dashboard (Streamlit)

- Carga artefactos de entrenamiento (historia, métricas, hiperparámetros).
- Muestra KPIs principales, curvas de entrenamiento y comparación de modelos.
- Usa un header tipo web app y un tema oscuro/claro inyectado vía CSS.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import REPORTS_DIR
from src.app.theme import inject_theme
from src.app.components.metric_card import metric_card
from src.app.components.training_charts import (
    training_loss_chart,
    training_metric_chart,
)
from src.app.components.header import app_header

# =======================
# Configuración de página
# =======================
st.set_page_config(
    page_title="ML Training Report",
    layout="wide",
    page_icon="📈",
)

# =======================
# Sidebar: tema y run
# =======================
mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
inject_theme(dark=(mode == "Dark"))

st.sidebar.markdown("### Run")
# TODO: selector de run cuando tengas múltiples ejecuciones
# run_id = st.sidebar.selectbox("Run ID", ["actual"])  # placeholder

# =======================
# Carga de artefactos
# =======================
metrics_path: Path = REPORTS_DIR / "metrics_nn.json"
hyperparams_path: Path = REPORTS_DIR / "hyperparams_nn.json"
history_path: Path = REPORTS_DIR / "history_nn.csv"
model_comp_path: Path = REPORTS_DIR / "model_comparison.csv"

try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    hyperparams = json.loads(hyperparams_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)
    model_comp = pd.read_csv(model_comp_path)
except FileNotFoundError as exc:
    st.error(f"No se encontró el archivo de reportes: {exc}")
    st.stop()

# =======================
# HEADER estilo dashboard
# =======================
run_ts = datetime.now()  # más adelante podrías leerlo de un artefacto

app_header(
    title="Predicción de Defectos en Procesos Industriales",
    subtitle="Zero-Inflated Neural Network",
    branch="main",
    run_timestamp=run_ts,
    live=True,
)

# Pequeño overview para que se sienta más “web page”
st.markdown(
    """
    <p class="ml-muted">
      Monitor de entrenamiento para el modelo Zero-Inflated Neural Network que predice el procentaje de defectos en la variable
      Und_2a_percentage. Incluye métricas, curvas de
      entrenamiento y comparación frente a otras modelos.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =======================
# ROW 1: metric cards
# =======================
col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("R² (val)", f"{metrics.get('R2', float('nan')):.3f}", "Zero-Inflated NN")

with col2:
    metric_card("RMSE (val)", f"{metrics.get('RMSE', float('nan')):.4f}", "Und_2a_percentage")

with col3:
    best_epoch = history["val_loss"].idxmin()
    metric_card("Best Epoch", str(best_epoch), "mínimo val_loss")

with col4:
    metric_card("Epochs", str(len(history)), "entrenamiento completado")

# =======================
# ROW 2: charts
# =======================
st.markdown("### Training Dynamics")

left, right = st.columns(2)

with left:
    st.subheader("Training & Validation Loss")
    training_loss_chart(history_path)

with right:
    st.subheader("Training & Validation RMSE")
    training_metric_chart(history_path, metric="rmse")

# =======================
# ROW 3: hyperparameters + model comparison
# =======================
st.markdown("### Experiment Details")

hp_col, table_col = st.columns([1, 2])

with hp_col:
    st.subheader("Hyperparameters")
    for k, v in hyperparams.items():
        st.markdown(f"- **{k}**: `{v}`")

with table_col:
    st.subheader("Model Comparison")

    best_idx = model_comp["R2"].idxmax()
    model_comp["status"] = "completed"
    model_comp.loc[best_idx, "status"] = "best"

    styled = model_comp.style.apply(
        lambda s: [
            (
                "background-color: rgba(16,185,129,0.12); color:#6ee7b7"
                if v == "best"
                else ""
            )
            for v in s
        ],
        subset=["status"],
    )

    st.dataframe(
        styled,
        width="stretch",  # reemplazo de use_container_width=True
    )
