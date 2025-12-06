# src/app/components/confusion_matrix.py
import streamlit as st
import numpy as np
import pandas as pd

def confusion_matrix_heatmap(cm: np.ndarray, labels=None):
    """
    Placeholder genérico; útil si en algún momento añades un modelo de clasificación.
    """
    st.subheader("Confusion Matrix")

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)
