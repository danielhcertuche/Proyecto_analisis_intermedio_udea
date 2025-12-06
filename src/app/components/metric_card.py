# src/app/components/metric_card.py
import streamlit as st

def metric_card(title: str, value: str, subtitle: str = "", delta: str | None = None):
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
        if subtitle or delta:
            sub = subtitle
            if delta:
                sub += f" · Δ {delta}"
            st.markdown(f'<div class="metric-sub">{sub}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
