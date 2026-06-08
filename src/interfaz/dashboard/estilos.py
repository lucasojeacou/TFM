"""CSS global del dashboard Streamlit."""

from __future__ import annotations

import streamlit as st


def cargar_estilos() -> None:
    """Inyecta el CSS global del dashboard en Streamlit."""
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.25rem; max-width: 1560px; }
            button[data-baseweb="tab"] { flex: 1; font-weight: 800 !important; font-size: 1.05rem !important; padding: 0.85rem 1rem !important; }
            button[data-baseweb="tab"] p { font-size: 1.05rem !important; }
            iframe { border: 1px solid rgba(148, 163, 184, 0.22) !important; border-radius: 8px; }
            div[data-testid="stMetricValue"] { font-size: 1.42rem; }
            .small-note { color: #64748b; font-size: 0.92rem; margin-top: -0.45rem; }
            .step-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; margin: 0.6rem 0 1rem 0; }
            .step-card { border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 8px; padding: 0.72rem 0.85rem; background: rgba(248, 250, 252, 0.72); min-height: 86px; }
            .step-card-active { border-color: #1d4ed8; background: #2563eb; color: #ffffff; box-shadow: 0 8px 22px rgba(37, 99, 235, 0.28); }
            .step-card-active .step-copy { color: #dbeafe; }
            .step-title { font-weight: 800; margin-bottom: 0.2rem; }
            .step-copy { color: #475569; font-size: 0.88rem; line-height: 1.28; }
            .status-pill { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.7rem; border-radius: 999px; font-weight: 800; font-size: 0.82rem; letter-spacing: 0.02em; text-transform: uppercase; margin-bottom: 0.6rem; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12); }
            .status-ok { background: linear-gradient(135deg, #22c55e, #16a34a); color: #ffffff; }
            .status-mid { background: linear-gradient(135deg, #fb923c, #f97316); color: #ffffff; }
            .status-bad { background: linear-gradient(135deg, #ef4444, #dc2626); color: #ffffff; }
            .status-none { background: linear-gradient(135deg, #64748b, #475569); color: #ffffff; }
            .metric-strip { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 0.7rem; margin: 0.7rem 0 1rem 0; }
            .metric-card { border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 8px; padding: 0.7rem 0.85rem; background: #ffffff; min-height: 76px; }
            .metric-label { color: #64748b; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
            .metric-value { color: #0f172a; font-size: 1.18rem; font-weight: 800; margin-top: 0.15rem; overflow-wrap: anywhere; }
            .phase-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-top: 0.5rem; }
            .phase-badge { border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-weight: 800; padding: 0.35rem 0.75rem; white-space: nowrap; }
            div[role="progressbar"] > div { background-color: #2563eb !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
