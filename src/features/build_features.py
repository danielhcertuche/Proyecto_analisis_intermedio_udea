import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import (
    TARGET_COL,
    TIME_COL,
    TEMPORAL_SPLIT_QUANT,
    OUTLIER_LOWER_QUANT,
    OUTLIER_UPPER_QUANT,
)

def temporal_train_test_split(df: pd.DataFrame):
    df = df.copy().sort_values(TIME_COL)

    split_value = df[TIME_COL].quantile(TEMPORAL_SPLIT_QUANT)

    train_df = df[df[TIME_COL] <= split_value]
    test_df  = df[df[TIME_COL] >  split_value]

    X_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]

    X_test  = test_df.drop(TARGET_COL, axis=1)
    y_test  = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test

def remove_target_outliers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    q_low  = y_train.quantile(OUTLIER_LOWER_QUANT)
    q_high = y_train.quantile(OUTLIER_UPPER_QUANT)
    mask = (y_train >= q_low) & (y_train <= q_high)
    return X_train[mask], y_train[mask]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )
