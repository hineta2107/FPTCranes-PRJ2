from __future__ import annotations

import os
import streamlit as st
from src.config import Config

# Tài khoản được hỗ trợ:
#   admin  / AIJob2026!  → toàn quyền xem tất cả tính năng
#   user   / user123     → ứng viên phổ thông, ẩn các trường đánh giá thị trường

_ACCOUNTS = {
    "admin": {"password": "AIJob2026!", "role": "admin"},
    "user":  {"password": "user123",    "role": "user"},
}


def require_login() -> None:
    if st.session_state.get("authenticated", False):
        return

    # Cho phép override qua env var (chỉ áp dụng cho tài khoản admin)
    cfg = Config.load()
    prediction = cfg.get("prediction", {})
    env_user = os.getenv("AI_JOB_USER")
    env_pass = os.getenv("AI_JOB_PASSWORD")

    st.markdown("## 🔐 AI Job Market Salary Prediction")
    st.caption("Authenticated technical dashboard · local academic demo")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        authenticated = False
        role = "user"

        # Kiểm tra env override cho admin
        if env_user and env_pass:
            if username == env_user and password == env_pass:
                authenticated = True
                role = "admin"
        # Kiểm tra danh sách tài khoản nội bộ
        if not authenticated and username in _ACCOUNTS:
            acct = _ACCOUNTS[username]
            if password == acct["password"]:
                authenticated = True
                role = acct["role"]

        if authenticated:
            st.session_state["authenticated"] = True
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info(
        "Demo credentials:\n"
        "- **admin** / AIJob2026! → full access\n"
        "- **user** / user123 → candidate view (market fields hidden)"
    )
    st.stop()
