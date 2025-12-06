from xgboost import XGBRegressor

def get_xgb_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
