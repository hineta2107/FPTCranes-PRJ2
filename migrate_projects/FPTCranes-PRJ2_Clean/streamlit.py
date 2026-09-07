from __future__ import annotations

import streamlit as st

from src.components._login import require_login
from src.components._styles import configure_page
from src.pages._nav import MENU
from src.pages import _best_model, _data_clean, _data_ready, _model_comparison, _pipeline, _prediction


def main() -> None:
    configure_page()
    require_login()
    with st.sidebar:
        st.markdown("### 💼 AI Job Market")
        st.caption("Salary Prediction · 12-stage technical dashboard")
        page = st.radio("Menu", MENU, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Data-quality-first · leakage-safe · temporal locked test · same-bundle serving")
        if st.button("Sign out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    if page == MENU[0]:
        _data_clean.render()
    elif page == MENU[1]:
        _data_ready.render()
    elif page == MENU[2]:
        _model_comparison.render()
    elif page == MENU[3]:
        _best_model.render()
    elif page == MENU[4]:
        _prediction.render()
    else:
        _pipeline.render()


if __name__ == "__main__":
    main()
