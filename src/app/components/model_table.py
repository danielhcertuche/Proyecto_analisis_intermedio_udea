# src/app/components/model_table.py
import streamlit as st
import pandas as pd

def model_comparison_table(df: pd.DataFrame):
    """
    Tabla comparativa de modelos (R2, RMSE, etc.).
    """
    if df is None or df.empty:
        st.info("Aún no hay resultados de comparación de modelos.")
        return

    st.subheader("Model Comparison")
    st.markdown('<div class="ml-table">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
