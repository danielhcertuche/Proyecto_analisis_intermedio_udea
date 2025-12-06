# src/app/streamlit_app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config.settings import REPORTS_DIR
from src.app.theme import inject_theme
from src.app.components.metric_card import metric_card
from src.app.components.training_charts import (
    training_loss_chart,
    training_metric_chart,
)

st.set_page_config(
    page_title="ML Training Report",
    layout="wide",
    page_icon="📈",
)

# --- sidebar: modo claro/oscuro y selección de run ---
mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
inject_theme(dark=(mode == "Dark"))

st.sidebar.markdown("### Run")
# si en un futuro guardas múltiples runs, aquí iría el selector

# --- carga artefactos ---
metrics_path = REPORTS_DIR / "metrics_nn.json"
hyperparams_path = REPORTS_DIR / "hyperparams_nn.json"
history_path = REPORTS_DIR / "history_nn.csv"
model_comp_path = REPORTS_DIR / "model_comparison.csv"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
hyperparams = json.loads(hyperparams_path.read_text(encoding="utf-8"))
history = pd.read_csv(history_path)
model_comp = pd.read_csv(model_comp_path)

# =======================
# HEADER
# =======================
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <div>
        <h2 style="margin-bottom:2px;">ML Training Report</h2>
        <span style="font-size:0.8rem;color:#9ca3af;">Defectos · Zero-Inflated Neural Network</span>
      </div>
      <div style="display:flex;gap:12px;align-items:center;">
        <span style="font-size:0.75rem;color:#9ca3af;">branch: main</span>
        <span style="background:#22c55e1a;border-radius:999px;padding:4px 10px;font-size:0.75rem;color:#6ee7b7;">
          ● Live
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =======================
# ROW 1: metric cards
# =======================
col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("R² (val)", f"{metrics['R2']:.3f}", "Zero-Inflated NN")
with col2:
    metric_card("RMSE (val)", f"{metrics['RMSE']:.4f}", "Und_2a_percentage")
with col3:
    metric_card("Best Epoch", str(history["val_loss"].idxmin()), "mínimo val_loss")
with col4:
    metric_card("Epochs", str(len(history)), "entrenamiento completado")

# =======================
# ROW 2: charts
# =======================
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
hp_col, table_col = st.columns([1, 2])

with hp_col:
    st.subheader("Hyperparameters")
    hp_items = list(hyperparams.items())
    for k, v in hp_items:
        st.markdown(f"- **{k}**: `{v}`")

with table_col:
    st.subheader("Model Comparison")
    # marcar mejor modelo por R2
    best_idx = model_comp["R2"].idxmax()
    model_comp["status"] = "completed"
    model_comp.loc[best_idx, "status"] = "best"

    st.dataframe(
        model_comp.style.apply(
            lambda s: [
                "background-color: rgba(16,185,129,0.12); color:#6ee7b7"
                if v == "best"
                else ""
                for v in s
            ],
            subset=["status"],
        ),
        use_container_width=True,
    )
