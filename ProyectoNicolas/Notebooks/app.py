import streamlit as st
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_xgb_und2a.pkl")

# ================================
# Cargar modelo
# ================================
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Predicción Und 2a", layout="centered")

st.title("Predicción de Porcentaje de Unidades 2a")
st.write("Ingresa los valores del proceso:")

# ================================
# FORMULARIO AUTOMÁTICO
# ================================
input_data = {}

for col in model.named_steps["preprocessor"].feature_names_in_:
    if col in ["Tipo_TEJ", "Tecnologia", "C", "MP", "mp_categoria"]:
        input_data[col] = st.text_input(col)
    else:
        input_data[col] = st.number_input(col, step=1.0)

# ================================
# Convertir a DataFrame
# ================================
input_df = pd.DataFrame([input_data])

# ================================
# Botón de predicción
# ================================
if st.button("Predecir porcentaje"):
    pred = model.predict(input_df)[0]

    st.subheader("Resultado")
    st.metric("Und_2a_percentage", f"{pred*100:.2f}%")

    if pred < 0.05:
        st.success("✅ Riesgo bajo de rechazo")
    elif pred < 0.12:
        st.warning("⚠️ Riesgo medio de rechazo")
    else:
        st.error("🔴 Riesgo alto de rechazo")
