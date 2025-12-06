from pathlib import Path
import pandas as pd

from src.config.settings import (
    RAW_DIR,
    PROCESSED_DIR,
    RAW_DATASET_NAME,
    CLEAN_DATASET_NAME,
)

def load_raw_dataset() -> pd.DataFrame:
    path = RAW_DIR / RAW_DATASET_NAME
    return pd.read_csv(path)

def load_clean_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / CLEAN_DATASET_NAME
    return pd.read_csv(path)
