from __future__ import annotations

import os
import streamlit as st
from src.config import Config


def require_login() -> None:
    if st.session_state.get("authenticated", False):
        return

    cfg = Config.load()
    prediction = cfg.get("prediction", {})
    expected_user = os.getenv("AI_JOB_USER", prediction.get("demo_username", "admin"))
    expected_password = os.getenv("AI_JOB_PASSWORD", prediction.get("demo_password", "AIJob2026!"))

    st.markdown("## 🔐 AI Job Market Salary Prediction")
    st.caption("Authenticated technical dashboard · local academic demo")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        if username == expected_user and password == expected_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.info("Local demo credentials are configured in `config/project.yaml`. Replace them before shared deployment.")
    st.stop()
