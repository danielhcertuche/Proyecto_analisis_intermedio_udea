# src/app/theme.py
import streamlit as st


def inject_theme(dark: bool = True) -> None:
    """
    Inyecta CSS para tema oscuro o claro.
    - dark=True  -> dashboard tipo ML nocturno
    - dark=False -> variante clara simplificada
    """
    if dark:
        bg = "#020617"          # fondo casi negro
        panel = "rgba(15,23,42,0.85)"
        text_main = "#e5e7eb"
        text_muted = "#9ca3af"
    else:
        bg = "#f3f4f6"          # gris claro
        panel = "rgba(255,255,255,0.97)"
        text_main = "#111827"
        text_muted = "#6b7280"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg-color: {bg};
            --panel-color: {panel};
            --card-color: {panel};
            --border-color: rgba(148, 163, 184, 0.35);
            --primary: #0ea5e9;
            --success: #10b981;
            --warning: #fbbf24;
            --text-main: {text_main};
            --text-muted: {text_muted};
            --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco,
                     Consolas, "Liberation Mono", "Courier New", monospace;
        }}

        /* Fondo general */
        .main, .stApp {{
            background-color: var(--bg-color) !important;
            color: var(--text-main);
        }}

        /* Contenedor principal más tipo web (centrado + ancho máximo) */
        .block-container {{
            max-width: 1200px;
            padding-top: 100px !important;
            padding-bottom: 3rem;
            margin-top: 80px !important;
            margin: 10px auto;
        }}

        /* Ajuste de títulos */
        h1, h2, h3, h4 {{
            color: var(--text-main);
        }}

        h2 {{
            font-size: 1.4rem !important;
            margin-bottom: 0.5rem !important;
        }}

        h3 {{
            font-size: 1.1rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.4rem !important;
        }}

        /* Cards KPI */
        .ml-card {{
            background: var(--card-color);
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid var(--border-color);
            box-shadow: 0 18px 35px rgba(15,23,42,0.35);
            transition: transform 0.18s ease-out, box-shadow 0.18s ease-out,
                        border-color 0.18s ease-out;
            margin-top: 0.15rem 
            margin-bottom: 1rem;
        }}

        .ml-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 45px rgba(15,23,42,0.55);
            border-color: rgba(56,189,248,0.6);
        }}

        .ml-kpi {{
            font-family: var(--mono);
            font-size: 1.9rem;
        }}

        .ml-kpi-sub {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        /* Badges */
        .ml-badge-success {{
            background: rgba(16, 185, 129, 0.12);
            color: #6ee7b7;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.7rem;
        }}

        .ml-badge-primary {{
            background: rgba(14, 165, 233, 0.12);
            color: #7dd3fc;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.7rem;
        }}

        /* Tablas */
        .ml-table {{
            border-radius: 18px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}

        .ml-muted {{
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 60px;
        }}

        /* Header tipo web app */
        .ml-header {{
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            z-index: 50;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            background: var(--panel-color);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 3px;
            width: 100%;
        }}

        .ml-header-inner {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
        }}

        .ml-header-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .ml-logo {{
            width: 40px;
            height: 40px;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--primary), var(--success));
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--mono);
            font-size: 1.1rem;
            color: white;
        }}

        .ml-header-title {{
            font-weight: 700;
            font-size: 1.15rem;
        }}

        .ml-header-subtitle {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .ml-header-right {{
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .ml-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 4px 10px;
            border: 1px solid var(--border-color);
            background: rgba(15,23,42,0.6);
            font-family: var(--mono);
        }}

        .ml-badge-live {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 4px 12px;
            background: rgba(16,185,129,0.12);
            border: 1px solid rgba(16,185,129,0.35);
            color: #6ee7b7;
            font-size: 0.75rem;
        }}

        .ml-badge-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34,197,94,0.35);
            animation: ml-pulse 1.4s ease-out infinite;
        }}


        /* Tarjetas KPI (fondo oscuro) */

        .ml-metric-card {{
        background: #020617;                /* casi negro, buen contraste */
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.35); /* gris azulado */
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        }}

        .ml-metric-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        }}

        .ml-metric-title {{
        font-size: 0.78rem;
        font-weight: 500;
        color: #9ca3af;                     /* gris claro, contraste OK */
        letter-spacing: 0.04em;
        }}

        .ml-metric-value {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #f9fafb;                     /* casi blanco */
        }}

        .ml-metric-sub {{
        font-size: 0.75rem;
        color: #6b7280;                     /* gris medio, aún legible */
        }}

        @keyframes ml-pulse {{
            0%   {{ transform: scale(0.9); opacity: 1; }}
            70%  {{ transform: scale(1.1); opacity: 0.3; }}
            100% {{ transform: scale(0.9); opacity: 0.8; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
