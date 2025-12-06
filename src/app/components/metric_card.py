# src/app/components/metric_card.py
import streamlit as st


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    delta: str | None = None,
):
    """
    Tarjeta de métrica tipo KPI para dashboards oscuros.

    - `title`: etiqueta de la métrica (ej. "R² (val)").
    - `value`: valor destacado (ej. "0.923").
    - `subtitle`: texto contextual (modelo, variable…).
    - `delta`: si lo envías, se agrega como Δ en el subtítulo.
    """
    with st.container():
        st.markdown(
            """
            <div class="ml-metric-card">
              <div class="ml-metric-header">
                <span class="ml-metric-title">{title}</span>
                <span class="ml-metric-value">{value}</span>
              </div>
              {subtitle_block}
            </div>
            """.format(
                title=title,
                value=value,
                subtitle_block=(
                    f'<div class="ml-metric-sub">{subtitle}'
                    + (f" · Δ {delta}" if delta else "")
                    + "</div>"
                    if (subtitle or delta)
                    else ""
                ),
            ),
            unsafe_allow_html=True,
        )
