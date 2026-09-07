from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str, icon: str = "analytics") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-icon">{icon}</div>
          <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_intro(stage: str, objective: str, input_text: str, output_text: str) -> None:
    st.markdown(f"### {stage}")
    st.caption(objective)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Input**  \n`{input_text}`")
    c2.markdown(f"**Output / evidence**  \n`{output_text}`")
