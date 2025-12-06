# src/app/components/hyperparams_card.py
import streamlit as st
import pandas as pd

def hyperparams_card(params: dict):
    """
    Render simple de hiperparámetros en formato grid.
    """
    if not params:
        st.info("No se encontraron hiperparámetros para este experimento.")
        return

    st.subheader("Hyperparameters")

    df = pd.DataFrame(
        [{"parameter": k, "value": v} for k, v in params.items()]
    )

    st.markdown('<div class="ml-card">', unsafe_allow_html=True)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
