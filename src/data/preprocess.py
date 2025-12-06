import pandas as pd
from src.config.settings import (
    RAW_DIR,
    PROCESSED_DIR,
    RAW_DATASET_NAME,
    CLEAN_DATASET_NAME,
)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["anio_mes"] = pd.to_datetime(data["anio_mes"])

    data["Und_1a"] = data["Und_1a"].fillna(0)
    data["Und_2a"] = data["Und_2a"].fillna(0)

    data["Tipo_2a"] = data["Tipo_2a"].fillna("Unknown")

    if "Reprogramado" in data.columns:
        data = data.drop("Reprogramado", axis=1)

    data.dropna(subset=["C", "MP", "mp_categoria"], inplace=True)

    data["Rechazo_comp"] = data["Rechazo_comp"].fillna(0)

    data["total_und"] = data["Und_1a"] + data["Und_2a"]
    data["Und_2a_percentage"] = data["Und_2a"] / data["total_und"]
    data["Und_2a_percentage"] = data["Und_2a_percentage"].fillna(0)

    columns_to_drop = [
        "Co_Dano",
        "Descr_Dano",
        "Gr_Dano_Dano",
        "Gr_Dano_Secc",
        "Tipo_2a",
        "anio_mes",
    ]
    existing = [c for c in columns_to_drop if c in data.columns]
    data = data.drop(columns=existing)

    if "Tecnologia" in data.columns:
        data["Tecnologia"] = data["Tecnologia"].fillna(data["Tecnologia"].mode()[0])

    if "Pas" in data.columns:
        data["Pas"] = data["Pas"].fillna(data["Pas"].median())

    if "rechazo_flag" in data.columns:
        data["rechazo_flag"] = data["rechazo_flag"].fillna(
            data["rechazo_flag"].mode()[0]
        )

    return data

def run_cleaning_pipeline() -> None:
    df_raw = pd.read_csv(RAW_DIR / RAW_DATASET_NAME)
    df_clean = clean_dataset(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DIR / CLEAN_DATASET_NAME, index=False)
