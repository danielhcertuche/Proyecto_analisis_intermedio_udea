from pathlib import Path

# raíz del proyecto = un nivel arriba de este archivo
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = DATA_DIR / "models"

REPORTS_DIR   = PROJECT_ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
INFORME_DIR   = REPORTS_DIR / "informe"

RANDOM_STATE  = 42

RAW_DATASET_NAME   = "dataset.csv"
CLEAN_DATASET_NAME = "dataset_cleaned.csv"
RAW_DATA_PATH = RAW_DIR / RAW_DATASET_NAME  

TARGET_COL_ORIGINAL = "Und_2a"
TARGET_COL           = "Und_2a_percentage"
TIME_COL             = "semana_anio"
TEMPORAL_SPLIT_QUANT = 0.8
OUTLIER_LOWER_QUANT  = 0.01
OUTLIER_UPPER_QUANT  = 0.99
