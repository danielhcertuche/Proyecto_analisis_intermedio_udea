# src/config/nn_config.py
"""
Configuración específica para el modelo NN tabular con embeddings.
"""

# Columnas con contribución despreciable (<0.05% permutation importance)
NOISE_COLS = ["Tecnologia", "Tur", "categoria_producto", "semana_anio", "g_art_id"]

# Definidas a partir del análisis de negocio + estructura de datos crudos
USER_NUMERICAL = [
    "semana_anio", "Tur", "planta_id", "seccion_id", "maq_id", "Pas",
    "producto_id", "estilo_id", "Tal", "Col", "Tal_Fert", "Col_Fert",
    "Componentes", "g_art_id", "mp_id", "Rechazo_comp",
    "rechazo_flag", "Tipo_2a_encoded",
]

USER_CATEGORICAL = [
    "Tipo_TEJ", "Tecnologia", "C", "categoria_producto", "MP", "mp_categoria",
]

# Variables con fuga de información respecto al target
LEAKAGE_COLS = ["Rechazo_comp", "rechazo_flag", "Tipo_2a_encoded"]

# Subdirectorio dentro de MODELS_DIR para la NN
NN_MODEL_SUBDIR = "nn_zero_inflated"
NN_KERAS_NAME = "modelo_defectos.keras"
NN_PIPELINE_PKL = "pipeline_completo.pkl"
