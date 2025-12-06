# src/app/components/training_charts.py
import streamlit as st
import pandas as pd

def training_loss_chart(history_path):
    hist = pd.read_csv(history_path)
    fig = st.line_chart(
        hist[["loss", "val_loss"]],
        height=260,
    )
    return fig

def training_metric_chart(history_path, metric="rmse"):
    hist = pd.read_csv(history_path)
    cols = [c for c in hist.columns if metric in c]
    st.line_chart(hist[cols], height=260)
