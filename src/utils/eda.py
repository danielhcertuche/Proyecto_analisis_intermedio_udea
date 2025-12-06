# src/utils/eda.py
import pandas as pd

def missing_percentage(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)

def get_feature_lists(df: pd.DataFrame, target_col: str):
    num = (
        df.select_dtypes(include=["int64", "float64"])
        .columns.drop(target_col, errors="ignore")
        .tolist()
    )
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat
