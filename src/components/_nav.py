from __future__ import annotations

from typing import Any

import streamlit as st


def build_navigation():
    pages = [
        st.Page(
            "src/pages/_overview.py",
            title="Executive overview",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "src/pages/_data_clean.py",
            title="1. Data basic clean",
            icon="🧹",
            url_path="data-clean",
        ),
        st.Page(
            "src/pages/_data_ready.py",
            title="2. Data ready for ML",
            icon="🧠",
            url_path="data-ready",
        ),
        st.Page(
            "src/pages/_model_comparison.py",
            title="3. Model comparison",
            icon="📊",
            url_path="model-comparison",
        ),
        st.Page(
            "src/pages/_best_model.py",
            title="4. Best model and importance",
            icon="🏆",
            url_path="best-model",
        ),
        st.Page(
            "src/pages/_prediction.py",
            title="5. Salary prediction",
            icon="💰",
            url_path="prediction",
        ),
        st.Page(
            "src/pages/_pipeline.py",
            title="12-stage pipeline",
            icon="🧭",
            url_path="pipeline",
        ),
        st.Page(
            "src/pages/_outputs.py",
            title="Project outputs",
            icon="📁",
            url_path="outputs",
        ),
    ]

    return st.navigation(pages, position="sidebar", expanded=True)


def render_sidebar_context(metadata: dict[str, Any]) -> None:
    with st.sidebar:
        st.caption("12-stage project dashboard")
        st.caption(f"Model: {metadata.get('model_name', '—')}")
        vocabulary_size = metadata.get("skill_vocabulary_count", "—")
        st.caption(f"Skill vocabulary: {vocabulary_size} tokens")
        if st.button("Log out", icon="🚪", width="stretch"):
            st.session_state.authenticated = False
            st.rerun()
