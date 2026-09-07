from __future__ import annotations

import streamlit as st
from src.config import Config


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Job Market Salary Prediction",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css_path = Config.ASSET_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
