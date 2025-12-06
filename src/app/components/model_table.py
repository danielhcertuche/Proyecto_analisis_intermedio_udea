# src/app/components/model_table.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def model_comparison_table(df: pd.DataFrame) -> None:
    """
    Tabla comparativa de modelos con narrativa técnica:
    - Resalta el modelo con mejor R² como candidato principal.
    - Muestra métricas clave de validación.
    """
    if df is None or df.empty:
        st.info("Aún no hay resultados de comparación de modelos.")
        return

    # Nos quedamos con una copia para no mutar el DataFrame original
    df = df.copy()

    # -----------------------------
    # Derivar columna de estado
    # -----------------------------
    # Si ya existe una columna "status", la respetamos.
    if "status" not in df.columns and "R2" in df.columns:
        best_idx = df["R2"].idxmax()
        df["status"] = "Completado"
        df.loc[best_idx, "status"] = "Mejor"
    elif "R2" in df.columns:
        best_idx = df["R2"].idxmax()
    else:
        best_idx = None

    # -----------------------------
    # Renombrar columnas a español
    # -----------------------------
    rename_map = {
        "model": "Modelo",
        "R2": "R² (val)",
        "RMSE": "RMSE (val)",
        "MSE": "MSE (val)",
        "MAE": "MAE (val)",
        "status": "Estado",
    }
    df_disp = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # -----------------------------
    # Formato numérico y estilos
    # -----------------------------
    format_dict = {}
    if "R² (val)" in df_disp.columns:
        format_dict["R² (val)"] = "{:.3f}"
    if "RMSE (val)" in df_disp.columns:
        format_dict["RMSE (val)"] = "{:.4f}"
    if "MSE (val)" in df_disp.columns:
        format_dict["MSE (val)"] = "{:.6f}"
    if "MAE (val)" in df_disp.columns:
        format_dict["MAE (val)"] = "{:.4f}"

    styled = df_disp.style.format(format_dict)

    # Píldora verde para el mejor modelo
    if "Estado" in df_disp.columns:
        def _style_estado(col):
            styles = []
            for v in col:
                if str(v).lower() in {"mejor", "best"}:
                    styles.append(
                        "background-color: rgba(16,185,129,0.15); "
                        "color: #a7f3d0; "
                        "border-radius: 999px; "
                        "text-align: center; "
                        "font-weight: 600;"
                    )
                else:
                    styles.append(
                        "background-color: rgba(148,163,184,0.12); "
                        "color: #e5e7eb; "
                        "border-radius: 999px; "
                        "text-align: center;"
                    )
            return styles

        styled = styled.apply(_style_estado, subset=["Estado"])

    # Fila ligeramente resaltada para el mejor R²
    if best_idx is not None:
        def _highlight_best(row):
            if row.name == best_idx:
                return [
                    "background-color: rgba(15,23,42,0.9); "
                    "border-bottom: 1px solid rgba(55,65,81,0.9); "
                    "font-weight: 600;"
                ] * len(row)
            return [
                "background-color: rgba(15,23,42,0.75); "
                "border-bottom: 1px solid rgba(31,41,55,0.9);"
            ] * len(row)

        styled = styled.apply(_highlight_best, axis=1)

    # -----------------------------
    # Render en Streamlit
    # -----------------------------
    st.markdown("### Comparación de modelos")
    st.markdown(
        """
        <p class="ml-muted">
        Rendimiento de las distintas modelos evaluados sobre el conjunto
        de validación. Se resalta el modelo con mejor R² como candidato
        principal para despliegue en producción.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ml-table">', unsafe_allow_html=True)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
