from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components._header import page_header
from src.config import Config
from src.pages._common import show_image

STAGES = [
    (1, "Load Data", "Fingerprint raw CSV and preserve source dimensions."),
    (2, "Project Scope & Initial Inspection", "Validate task, target, schema, dtypes and descriptive statistics."),
    (3, "Data Quality Check", "Audit nulls, duplicates, hidden tokens, cardinality and categorical values."),
    (4, "Corrupted Row Removal", "Remove confirmed header-token corruption; create canonical clean data."),
    (5, "Contradictory-Feature Investigation", "Audit experience inconsistency and target-adjacent salary metadata."),
    (6, "Feature Selection & Leakage Prevention", "Define explicit ALLOW/BLOCK policy and ablation sets."),
    (7, "Correlation Encoding & Analysis", "DEV-only one-hot/ordinal/multi-hot, RobustScaler diagnostics, correlation/VIF."),
    (8, "Train-Test Split", "Reserve 2026-03 locked test; fit preprocessing on TRAIN/DEV only."),
    (9, "Model Training & Comparison", "Dummy + five candidates on 5 expanding-window temporal folds."),
    (10, "Best Model & Importance Review", "Tune selected family, open locked test once, inspect importance/errors."),
    (11, "Save Deployable Pipeline + Metadata", "Serialize full bundle and pass reload-equivalence gate."),
    (12, "Streamlit Salary Prediction Dashboard", "Authenticated inference with same bundle, interval and review context."),
]


def render() -> None:
    page_header("12-stage pipeline", "One governed path from raw ingestion to leakage-safe temporal evaluation and deployable Streamlit inference", "🧭")
    show_image(Config.ASSET_DIR / "12_stage_pipeline.png")
    table = pd.DataFrame(STAGES, columns=["Stage", "Name", "Operational purpose"])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.info("Core governance: March-2026 is locked before target-aware diagnostics; preprocessors and skill vocabulary are fitted only on development/fold training data; model selection uses temporal CV; Stage 12 loads the same Stage-11 bundle.")
