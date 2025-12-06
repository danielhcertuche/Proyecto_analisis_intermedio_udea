# src/app/components/header.py
from __future__ import annotations

from datetime import datetime

import streamlit as st


def app_header(
    title: str,
    subtitle: str,
    branch: str = "main",
    run_timestamp: datetime | str | None = None,
    live: bool = True,
    logo_text: str = "ML",
) -> None:
    """
    Header tipo dashboard ML construido con HTML + clases CSS inyectadas en theme.py.

    - Título y subtítulo del experimento.
    - Rama de git.
    - Timestamp del último run.
    - Badge opcional "Live".
    """

    # --- formateo de timestamp ---
    if run_timestamp is None:
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elif isinstance(run_timestamp, datetime):
        ts_str = run_timestamp.strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = str(run_timestamp)

    live_badge_html = ""
    if live:
        live_badge_html = '<div class="ml-badge-live"><span class="ml-badge-dot"></span><span>Live</span></div>'

    st.markdown(
        f"""
        <header class="ml-header">
          <div class="ml-header-inner">
            <div class="ml-header-left">
              <div class="ml-logo">{logo_text}</div>
              <div>
                <div class="ml-header-title">{title}</div>
                <div class="ml-header-subtitle">{subtitle}</div>
              </div>
            </div>
            <div class="ml-header-right">
              <div class="ml-pill">
                <span>branch</span>
                <span>{branch}</span>
              </div>
              <div class="ml-pill">
                <span>last run</span>
                <span>{ts_str}</span>
              </div>
              {live_badge_html}
            </div>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )
