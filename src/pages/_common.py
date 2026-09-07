from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import streamlit as st


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Cannot read {path.name}: {exc}")
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Cannot read {path.name}: {exc}")
        return {}


def fmt_money(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "—"


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart not generated yet: `{path.name}`")


def evidence(title: str, finding: str, interpretation: str, action: str, kind: str = "info") -> None:
    icons = {"good": "✅", "warn": "⚠️", "risk": "🚨", "info": "ℹ️"}
    st.markdown(f"#### {icons.get(kind, 'ℹ️')} {title}")
    st.markdown(
        f"""
        <div class="evidence-card {kind}">
        <b>Observed evidence</b><br>{finding}<br><br>
        <b>Technical interpretation</b><br>{interpretation}<br><br>
        <b>Pipeline action</b><br>{action}
        </div>
        """,
        unsafe_allow_html=True,
    )


def guard(df: pd.DataFrame, message: str) -> bool:
    if df.empty:
        st.warning(message)
        return True
    return False
