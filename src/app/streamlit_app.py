import streamlit as st

from src.models.inference import predict_one
from src.config.settings import TARGET_COL

st.set_page_config(page_title="Modelo Und_2a", layout="centered")
st.title("Predicción de Und_2a_percentage (XGBoost)")

with st.form("prediction_form"):
    C            = st.text_input("C (código cliente)", "")
    MP           = st.text_input("MP (materia prima)", "")
    mp_categoria = st.text_input("Categoría MP", "")
    Tecnologia   = st.text_input("Tecnología", "")
    Pas          = st.number_input("Pas", value=0.0)
    semana_anio  = st.number_input("Semana del año", value=1, min_value=1, max_value=53)

    submitted = st.form_submit_button("Predecir")

if submitted:
    row = {
        "C": C,
        "MP": MP,
        "mp_categoria": mp_categoria,
        "Tecnologia": Tecnologia,
        "Pas": Pas,
        "semana_anio": semana_anio,
    }
    pred = predict_one(row)
    st.success(f"{TARGET_COL}: {pred:.4f}")
