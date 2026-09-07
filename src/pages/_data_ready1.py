from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, guard, read_csv, read_json, show_image

READY = Config.OUTPUT_DIR / "02_data_ready_for_machine_learning"


def render() -> None:
    page_header("2. Data ready for ML", "Stages 06–08 · leakage governance → DEV-only encoding/correlation → temporal locked-test readiness", "🧪")
    tabs = st.tabs(["06 · Feature Governance", "07 · Encoding & Analysis", "08 · Temporal Split"])

    with tabs[0]:
        stage_intro("Stage 06 · Feature Selection & Leakage Prevention", "Define an explicit ALLOW/BLOCK policy and ablation plan before model fitting.", "Basic-clean features", "Feature policy + leakage gate + ablation plan")
        policy = read_csv(READY / "06_feature_policy.csv")
        primary = read_csv(READY / "06_primary_model_features.csv")
        ablation = read_csv(READY / "06_ablation_plan.csv")
        gate = read_json(READY / "06_leakage_gate.json")
        if guard(policy, "Stage 06 outputs are missing."):
            return
        cols = st.columns(4)
        cols[0].metric("Serving fields", len(primary))
        cols[1].metric("Blocked fields", int(policy["policy"].eq("BLOCK").sum()))
        cols[2].metric("Phase-2 skills", int(policy["role"].astype(str).str.contains("PHASE2", case=False).sum()))
        cols[3].metric("Leakage gate", gate.get("status", "—"))
        left, right = st.columns([1, 1.35])
        with left:
            show_image(READY / "06_feature_governance_mix.png")
        with right:
            st.markdown("#### Primary serving feature set")
            st.dataframe(primary, use_container_width=True, hide_index=True, height=400)
        st.markdown("#### Full feature / leakage contract")
        st.dataframe(policy, use_container_width=True, hide_index=True, height=440)
        with st.expander("Required feature-family ablation plan", expanded=True):
            st.dataframe(ablation, use_container_width=True, hide_index=True)
        evidence("Leakage-control decision", f"{len(primary)} model-serving fields are admitted and {int(policy['policy'].eq('BLOCK').sum())} fields are blocked.", "Eligibility is based on semantics, inference-time availability, target derivation, identifier behavior and temporal stability—not correlation alone.", "Use the governed primary set for model development; quantify experience and skills through explicit ablation rather than silent inclusion/exclusion.", "good")

    with tabs[1]:
        stage_intro("Stage 07 · Correlation Encoding & Analysis", "Fit analytical preprocessing on DEV only; audit Pearson/Spearman association, VIF, pairwise redundancy and skills.", "TRAIN/DEV only", "Encoded diagnostics + correlations + VIF + 93-token skill evidence")
        corr = read_csv(READY / "07_target_correlations_dev.csv")
        vif = read_csv(READY / "07_vif_dev_numeric.csv")
        skills = read_csv(READY / "07_skill_frequency_and_salary_dev.csv")
        under50 = read_csv(READY / "07_skills_under_50_records_dev.csv")
        transform = read_csv(READY / "07_target_transform_review.csv")
        if guard(corr, "Stage 07 outputs are missing."):
            return
        strongest = corr.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Encoded features", len(corr))
        cols[1].metric("Skill vocabulary", len(skills))
        cols[2].metric("Strongest |Pearson|", f"{float(strongest['abs_pearson']):.3f}")
        cols[3].metric("VIF > 10", int((vif["vif"] > 10).sum()) if not vif.empty else 0)
        cols[4].metric("Skills <50 rows", len(under50))
        show_image(READY / "07_top30_correlations.png")
        left, right = st.columns(2)
        with left:
            show_image(READY / "07_pearson_vs_spearman.png")
        with right:
            show_image(READY / "07_vif_numeric.png")
        show_image(READY / "07_top30_skills.png")
        with st.expander("Target transformation review"):
            st.dataframe(transform, use_container_width=True, hide_index=True)
        with st.expander("All DEV-only correlations"):
            st.dataframe(corr, use_container_width=True, hide_index=True, height=450)
        with st.expander(f"Skills with <50 DEV records ({len(under50)})"):
            st.dataframe(under50, use_container_width=True, hide_index=True, height=380)
        evidence("Stage 07 interpretation", f"The strongest encoded association is `{strongest['feature']}` (Pearson {float(strongest['pearson']):+.3f}; Spearman {float(strongest['spearman']):+.3f}). The skill vocabulary contains {len(skills)} DEV-only tokens.", "Correlation describes association in this dataset, not causality. Pearson/Spearman divergence can reveal monotonic nonlinearity; VIF and pairwise checks diagnose redundancy. Skills are tokenized and de-duplicated rather than one-hot encoding whole strings.", "Carry the fitted preprocessing contract forward; all future validation/test rows are transformed only with DEV/fold-fitted encoders and skill vocabularies.", "info")

    with tabs[2]:
        stage_intro("Stage 08 · Train-Test Split", "Reserve March-2026 as locked final test and freeze preprocessing on TRAIN/DEV only.", "1,499 clean rows", "1,201 DEV + 298 locked TEST + encoded matrices")
        split = read_csv(READY / "08_split_summary.csv")
        before_after = read_csv(READY / "08_before_after_processing.csv")
        readiness = read_json(READY / "08_training_readiness.json")
        monthly = read_csv(READY / "08_monthly_distribution.csv")
        if guard(split, "Stage 08 outputs are missing."):
            return
        dev_row = split.loc[split["partition"].eq("TRAIN_DEV")].iloc[0]
        test_row = split.loc[split["partition"].eq("LOCKED_TEST")].iloc[0]
        cols = st.columns(4)
        cols[0].metric("TRAIN / DEV", int(dev_row["rows"]))
        cols[1].metric("Locked TEST", int(test_row["rows"]))
        cols[2].metric("Encoded features", readiness.get("encoded_feature_count", "—"))
        cols[3].metric("Skill vocab", readiness.get("skill_vocabulary_count", "—"))
        show_image(READY / "08_temporal_split_timeline.png")
        left, right = st.columns([1, 1.5])
        with left:
            fig = px.pie(split, names="partition", values="rows", hole=.58, title="Development vs locked test")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(before_after, use_container_width=True, hide_index=True, height=410)
        with st.expander("Monthly distribution"):
            st.dataframe(monthly, use_container_width=True, hide_index=True)
        evidence("Locked-test governance", f"DEV={int(dev_row['rows']):,} rows; March-2026 locked TEST={int(test_row['rows']):,} rows. Preprocessing fit scope: {readiness.get('fit_scope', 'TRAIN/DEV only')}.", "The final month is not used for feature selection, hyperparameter tuning or temporal CV, preventing look-ahead bias.", "Use expanding-window temporal CV inside DEV for Stage 09. Open locked-test labels only once after model family, tuning policy and narrative are frozen.", "good")
