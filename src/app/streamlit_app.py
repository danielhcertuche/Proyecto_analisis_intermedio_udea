# src/app/streamlit_app.py
"""
ML Training Report Dashboard (Streamlit)

- Carga artefactos de entrenamiento (historia, métricas, hiperparámetros).
- Muestra KPIs principales, curvas de entrenamiento y comparación de modelos.
- Usa un header tipo web app y un tema oscuro/claro inyectado vía CSS.
- Incluye una sección de predicción rápida (what-if) usando el modelo entrenado.
"""
import sys
from pathlib import Path

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st




# === AÑADE ESTO ANTES DE "from src.config.settings ..." ===
# Buscar la raíz del proyecto (la carpeta que contiene "src")
cwd = Path(__file__).resolve()
PROJECT_ROOT = None

for parent in [cwd, *cwd.parents]:
    if (parent / "src").is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError("No se encontró carpeta 'src' en los padres.")

# Añadir la raíz del proyecto al PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ahora sí, los imports de tu proyecto
from src.config.settings import REPORTS_DIR, TARGET_COL
from src.config.nn_config import NN_MODEL_SUBDIR, NN_KERAS_NAME, NN_PIPELINE_PKL


from src.config.settings import REPORTS_DIR, TARGET_COL
from src.app.theme import inject_theme
from src.app.components.metric_card import metric_card
from src.app.components.training_charts import (
    training_loss_chart,
    training_metric_chart,
)
from src.app.components.header import app_header
from src.app.components.quick_predict import quick_prediction_card

import logging

logging.getLogger("tornado.application").setLevel(logging.ERROR)
logging.getLogger("tornado.general").setLevel(logging.ERROR)


# --- NUEVO: imports para inferencia ---
from src.models.nn_inference import (
    load_nn_zero_inflated_bundle,
    predict_und_2a_from_raw,
)
from src.data.load_data import load_clean_dataset
from src.features.nn_features import reorganize_features_final

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
# Carga de artefactos de entrenamiento
# =======================
metrics_path: Path = REPORTS_DIR / "metrics_nn.json"
hyperparams_path: Path = REPORTS_DIR / "hyperparams_nn.json"
history_path: Path = REPORTS_DIR / "history_nn.csv"
model_comp_path: Path = REPORTS_DIR / "model_comparison.csv"


try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    hyperparams = json.loads(hyperparams_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)

    try:
        # intento normal
        model_comp = pd.read_csv(model_comp_path)
    except pd.errors.ParserError:
        # fallback si el CSV viene “sucio”
        model_comp = pd.read_csv(model_comp_path, on_bad_lines="skip")

        # normalizar nombres si viene de notebooks antiguos
        if "modelo" in model_comp.columns and "model" not in model_comp.columns:
            model_comp = model_comp.rename(columns={"modelo": "model"})

except FileNotFoundError as exc:
    st.error(f"No se encontró el archivo de reportes: {exc}")
    st.stop()

# =======================
# Cargar modelo + artefactos para inferencia (cacheado)
# =======================
@st.cache_resource
def get_nn_bundle():
    return load_nn_zero_inflated_bundle()


model, artefacts = get_nn_bundle()
embed_cols = artefacts["embed_cols"]
num_cols = artefacts["num_cols"]

# Dataset base para sugerir opciones por defecto en el formulario
df = load_clean_dataset()
X = df.drop(columns=[TARGET_COL])
X_clean, _, _ = reorganize_features_final(X)

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
      Monitor de entrenamiento para el modelo Zero-Inflated Neural Network que predice
      el porcentaje de defectos en la variable <code>Und_2a_percentage</code>.
      Incluye métricas, curvas de entrenamiento, comparación frente a otros modelos
      y una herramienta de predicción rápida tipo <em>what-if</em>.
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
    metric_card(
        "R² (val)",
        f"{metrics.get('R2', float('nan')):.3f}",
        "Zero-Inflated NN",
    )

with col2:
    metric_card(
        "RMSE (val)",
        f"{metrics.get('RMSE', float('nan')):.4f}",
        "Und_2a_percentage",
    )

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
# =======================
# ROW 4: QUICK PREDICTION (componente)
# =======================
quick_prediction_card(
    X_clean=X_clean,
    embed_cols=embed_cols,
    num_cols=num_cols,
    model=model,
    artefacts=artefacts,
)