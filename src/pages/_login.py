from __future__ import annotations

import hmac
import os

from rich.console import Console

import streamlit as st
from src.utils.auth import password_hash, resolve_auth_settings

logger = Console()

def require_authentication() -> None:
    """Stop the current rerun until the user authenticates."""
    st.session_state.setdefault("authenticated", False)
    if st.session_state.authenticated:
        return

    try:
        secrets = st.secrets
    except (FileNotFoundError, RuntimeError):
        secrets = {}
    settings = resolve_auth_settings(os.environ, secrets)

    st.title("🔐 AI job market salary prediction", anchor=False)
    st.caption(
        "Technical dashboard, model comparison, and interactive salary prediction."
    )

    _, center, _ = st.columns([1, 1.2, 1])
    with center.container(border=True):
        st.subheader("Log in", anchor=False)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter password"
            )
            submitted = st.form_submit_button(
                "Log in",
                type="primary",
                icon=":material/login:",
                width="stretch",
            )

        if settings.demo_mode:
            st.info(
                "Local demo login: ** admin / AIJob2026! **. Configure secrets or environment "
                "variables before shared deployment.",
                icon=":material/info:",
            )
        logger.log("login: user=%s, hash=%s", username, password)
        if submitted:
            username_matches = hmac.compare_digest(username, settings.username)
            password_matches = hmac.compare_digest(
                password_hash(password), settings.password_sha256
            )
            if username_matches and password_matches:
                st.session_state.authenticated = True
                st.rerun()
            st.error("Invalid username or password.", icon=":material/error:")

    st.stop()
