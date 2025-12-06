import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")


TARGET_COL = "Und_2a"

# Features numéricas y categóricas 
NUM_FEATURES = [
    "anio_mes",
    "semana_anio",
    "Tur",
    "planta_id",
    "seccion_id",
    "maq_id",
    "Pas",
    "producto_id",
    "estilo_id",
    "Tal",
    "Col",
    "Tal_Fert",
    "Col_Fert",
    "Componentes",
    "g_art_id",
    "mp_id",
    "Co_Dano",
    "Und_1a",
    "Rechazo_comp",
    "rechazo_flag",  
]

CAT_FEATURES = [
    "Tipo_TEJ",
    "Tecnologia",
    "C",
    "categoria_producto",
    "MP",
    "mp_categoria",
    "Descr_Dano",
    "Gr_Dano_Dano",
    "Gr_Dano_Secc",
    "Tipo_2a",
    "Reprogramado",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5