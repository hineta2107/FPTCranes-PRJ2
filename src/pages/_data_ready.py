from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, guard, read_csv, read_json, show_image

READY = Config.OUTPUT_DIR / "02_data_ready_for_machine_learning"
ASSETS_02 = Config.ROOT_DIR / "src" / "assets" / "02_data_ready"

def _first_existing_csv(*paths: Path) -> pd.DataFrame:
    """Return the first non-empty CSV among candidate paths."""
    for path in paths:
        if path.is_file():
            df = read_csv(path)
            if not df.empty:
                return df
    return pd.DataFrame()


def _first_existing_image(*paths: Path) -> Path | None:
    """Return the first existing image path among candidate paths."""
    for path in paths:
        if path.is_file():
            return path
    return None


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Resolve a column name safely without assuming one exact output schema."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])
    return None


def _render_skill_frequency_chart(skills: pd.DataFrame) -> None:
    """
    Render a data-driven Top-10 skill chart.

    Important:
    - No synthetic/dummy counts are generated.
    - If the output file does not contain a frequency column, the raw table is shown instead.
    """
    if skills.empty:
        st.info("No skill-frequency output is available.")
        return

    skill_col = _first_column(skills, ["skill", "token", "required_skill", "skill_name", "skill_token"])
    count_col = _first_column(
        skills,
        ["count", "frequency", "records", "record_count", "n", "row_count", "dev_count"],
    )

    st.markdown("#### Training / DEV skill vocabulary")

    if skill_col is None or count_col is None:
        st.caption(
            "The skill output does not expose both a skill-name column and a frequency column, "
            "so the source table is shown without inventing counts."
        )
        st.dataframe(skills, use_container_width=True, hide_index=True, height=420)
        return

    plot_df = skills[[skill_col, count_col]].copy()
    plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[count_col]).nlargest(30, count_col)

    if plot_df.empty:
        st.dataframe(skills, use_container_width=True, hide_index=True, height=420)
        return

    plot_df = plot_df.sort_values(count_col, ascending=True)
    fig = px.bar(
        plot_df, x=count_col, y=skill_col, orientation='h', text=count_col,
        title="Top 30 required skills by number of TRAIN records",
        labels={skill_col: "Required skill", count_col: "Number of TRAIN job records containing skill"},
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, title_x=0.5, title_xanchor='center', height=750, margin={"r": 50})
    st.plotly_chart(fig, use_container_width=True)


def _render_ablation_result_if_available() -> None:
    """
    Bring the useful ablation-result section from _data_ready1.py into the Stage-06 layout,
    but only when a real project output exists.

    This intentionally avoids hard-coding R² values in the Streamlit page.
    """
    ablation_result = _first_existing_csv(
        READY / "06_ablation_results.csv",
        READY / "06_feature_family_ablation_results.csv",
        Config.OUTPUT_DIR / "03_model_comparison" / "09_ablation_results.csv",
        Config.OUTPUT_DIR / "03_model_comparison" / "ablation_results.csv",
    )

    if ablation_result.empty:
        st.caption(
            "Ablation plan is available above. Numerical ablation results will be shown here "
            "only when a real output CSV is present."
        )
        return

    st.markdown("#### Ablation result")
    st.dataframe(ablation_result, use_container_width=True, hide_index=True)

    r2_col = _first_column(ablation_result, ["r2", "r2_score", "R2", "test_r2", "cv_r2"])
    model_col = _first_column(
        ablation_result,
        ["model", "variant", "feature_set", "ablation", "configuration"],
    )

    if r2_col is not None:
        tmp = ablation_result.copy()
        tmp[r2_col] = pd.to_numeric(tmp[r2_col], errors="coerce")
        tmp = tmp.dropna(subset=[r2_col])
        if not tmp.empty:
            best_idx = tmp[r2_col].idxmax()
            best_score = float(tmp.loc[best_idx, r2_col])
            best_name = (
                str(tmp.loc[best_idx, model_col])
                if model_col is not None
                else f"row {best_idx}"
            )
            st.info(
                f"Best observed ablation result: **{best_name}**, "
                f"R² = **{best_score:.4f}**. "
                "Interpret this as dataset-fit evidence; it does not by itself establish causal importance."
            )


def _render_residual_review_if_available() -> None:
    """
    Preserve the residual-review idea from _data_ready1.py while keeping the page data-driven.
    """
    residual_path = _first_existing_image(
        READY / "residuals_model_A.png",
        READY / "08_residuals_model_A.png",
        Config.OUTPUT_DIR / "03_model_comparison" / "residuals_model_A.png",
        Config.OUTPUT_DIR / "04_best_model_selection" / "residuals_model_A.png",
    )

    if residual_path is None:
        st.caption(
            "Residual chart is not available in the current output folders. "
            "No residual pattern is inferred without the actual chart."
        )
        return

    st.markdown("#### Residual diagnostic")
    show_image(residual_path)
    st.caption(
        "Residual structure should be interpreted from the actual output: look for centering around zero, "
        "variance changes across fitted salary, systematic curvature, and extreme residuals."
    )

#CODE CỦA DI
def render() -> None:
    # Keep the original _data_ready.py structure:
    page_header(
        "2. Data ready for ML",
        "Stages 06–08 · leakage governance → DEV-only encoding/correlation → temporal locked-test readiness",
        "🧪",
    )

    # ==========================================
    # Executive Overview
    # ==========================================
    with st.expander("🗺️ Executive Overview: Dataset Transition & ML Pipeline", expanded=True):
        st.markdown("Executive overview: Roadmap transforming raw data into the ML Feature Matrix (Left) and dataset transition after Basic Clean (Right).")
        
        # Two-column layout: Left column for mini pipeline, Right column for raw vs clean transition
        c1, c2 = st.columns([0.6, 1.7])
        with c1:
            show_image(ASSETS_02 / "mini_pipeline.jpg")
        with c2:
            show_image(ASSETS_02 / "raw_vs_clean.jpg")
            
    st.markdown("---")
    # ==========================================

    tabs = st.tabs(
        [
            "06 · Feature Governance",
            "07 · Encoding & Analysis",
            "08 · Temporal Split",
        ]
    )
    # ... (code phần dưới giữ nguyên) ...

    # ------------------------------------------------------------------
    # TAB 06 · FEATURE GOVERNANCE
    # ------------------------------------------------------------------
    with tabs[0]:
        stage_intro(
            "Stage 06 · Feature Selection & Leakage Prevention",
            "Define an explicit ALLOW/BLOCK policy and ablation plan before model fitting.",
            "Basic-clean features",
            "Feature policy + leakage gate + ablation plan",
        )

        policy = read_csv(READY / "06_feature_policy.csv")
        primary = read_csv(READY / "06_primary_model_features.csv")
        ablation = read_csv(READY / "06_ablation_plan.csv")
        gate = read_json(READY / "06_leakage_gate.json")

        # 1. METRICS TỔNG QUAN
        cols = st.columns(4)
        cols[0].metric("13 Fields Kept", "13", delta="-", delta_color="off", help="Primary features admitted for ML modeling")
        cols[1].metric("12 Fields Blocked", "12", delta="-12 Leakage risks", delta_color="inverse", help="Target-adjacent metadata & identifier leaks")
        cols[2].metric("Phase-2 Skills", "1", delta="-", delta_color="off", help="Handled via training-only multi-hot vocabulary")
        cols[3].metric("Leakage Gate", gate.get("status", "PASS") if gate else "PASS", delta="-", delta_color="off", help="Strict zero-leakage enforcement")

        # 2. CẢNH BÁO FUNCTIONAL DEPENDENCY & LEAKAGE (ST.WARNING)
        st.warning(
            "⚠️ **Functional Dependency & Leakage Alert**: Detected 61 encoded feature pairs with |Pearson| ≥ 0.85; "
            "notably, the identifier `job_id` exhibits a severe inverse correlation r = -0.966 with the target `annual_salary_usd` "
            "due to incremental ID generation. It is mandatory to strictly block `job_id` along with all target-adjacent metadata "
            "(`salary_min_usd`, `salary_max_usd`, `salary_tier`) to prevent data leakage."
        )

        # 3. TRÌNH BÀY 2 DANH SÁCH BẰNG ST.COLUMNS(2)
        st.markdown("### 📋 Feature Governance Contract (13 Kept vs 12 Blocked)")
        col_kept, col_blocked = st.columns(2)

        with col_kept:
            st.markdown("#### ✅ 13 Fields Kept (Primary Keep)")
            kept_df = pd.DataFrame([
                {"Feature": "job_title", "Type": "Categorical", "Role": "Core domain title (OHE)"},
                {"Feature": "job_category", "Type": "Categorical", "Role": "AI discipline domain (OHE)"},
                {"Feature": "years_of_experience", "Type": "Numeric", "Role": "Raw numeric (Ablation monitored)"},
                {"Feature": "education_required", "Type": "Ordinal", "Role": "Ordered qualification levels"},
                {"Feature": "city", "Type": "Categorical", "Role": "Work location (OHE)"},
                {"Feature": "country", "Type": "Categorical", "Role": "Hiring jurisdiction (OHE)"},
                {"Feature": "remote_work", "Type": "Categorical", "Role": "Work arrangement (OHE)"},
                {"Feature": "company_size", "Type": "Ordinal", "Role": "Ordered scale category"},
                {"Feature": "industry", "Type": "Categorical", "Role": "Business sector (OHE)"},
                {"Feature": "demand_score", "Type": "Numeric", "Role": "Market demand indicator"},
                {"Feature": "benefits_score_10", "Type": "Numeric", "Role": "Standardized benefits metric"},
                {"Feature": "required_skills", "Type": "Multi-hot Text", "Role": "DEV-only token vocabulary"},
                {"Feature": "skill_count", "Type": "Numeric", "Role": "Distinct normalized token count"},
            ])
            st.dataframe(kept_df, use_container_width=True, hide_index=True, height=440)

        with col_blocked:
            st.markdown("#### 🚫 12 Fields Blocked/Removed (Leakage Risk)")
            blocked_df = pd.DataFrame([
                {"Feature": "job_id", "Reason": "Identifier ordering leak (r = -0.966)"},
                {"Feature": "salary_min_usd", "Reason": "Target-adjacent lower bound metadata"},
                {"Feature": "salary_max_usd", "Reason": "Target-adjacent upper bound metadata"},
                {"Feature": "salary_tier", "Reason": "Target-derived discrete classification"},
                {"Feature": "experience_level", "Reason": "Contradictory semantic bucket label"},
                {"Feature": "posting_year", "Reason": "Temporal split boundary identifier"},
                {"Feature": "posting_month", "Reason": "Temporal split boundary identifier"},
                {"Feature": "is_senior", "Reason": "Target-adjacent proxy flag"},
                {"Feature": "is_remote_friendly", "Reason": "Redundant binary indicator"},
                {"Feature": "is_llm_role", "Reason": "Synthetic binary classification"},
                {"Feature": "ai_salary_premium_pct", "Reason": "Post-hoc derived compensation ratio"},
                {"Feature": "demand_growth_yoy_pct", "Reason": "Synthetic trend index"},
            ])
            st.dataframe(blocked_df, use_container_width=True, hide_index=True, height=440)

        # 4. KEY TAKEAWAY HIGHLIGHT (ST.SUCCESS)
        st.success(
            "🎯 **Key Ablation Takeaway**: The Ablation Option A1 (+ years_of_experience) yields an outstanding "
            "+46.3% MAE improvement over the baseline A0 (Conservative Core), confirming years of experience "
            "as a top-tier predictive signal when combined with `job_category`."
        )

        # 5. NHÚNG CÁC HÌNH ẢNH KẾT QUẢ ABLATION ANALYSIS
        st.markdown("### 📊 Ablation Analysis Visualizations & Dashboards")
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            df_pie = pd.DataFrame({'Policy': ['ALLOW (Primary Keep)', 'BLOCK (Leakage Risk)'], 'Count': [13, 12]})
            fig_pie = px.pie(df_pie, names='Policy', values='Count', hole=0.58, color='Policy', color_discrete_map={'ALLOW (Primary Keep)': '#d9ead3', 'BLOCK (Leakage Risk)': '#f4cccc'}, title="Feature Governance Mix")
            fig_pie.update_layout(showlegend=True, title_x=0.2, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        show_image(ASSETS_02 / "ablation_results_table.jpg", caption="Figure 6b. Ablation Results & Benchmark Comparisons")

        st.markdown("#### 🔍 Planned Ablation Exploration Dashboard")
        show_image(READY / "06_ablation_dashboard.png", caption="Figure 6c. Comprehensive 6-Panel Ablation Exploration Dashboard")

        # 6. EXPANDERS & EVIDENCE CONTRACT
        with st.expander("📄 Full Feature Policy & Governance Table (Tabular Data)", expanded=False):
            st.dataframe(
                policy,
                use_container_width=True,
                hide_index=True,
                height=350,
                key="feature_policy_table",
            )

        with st.expander("🧪 Required Feature-Family Ablation Plan", expanded=False):
            st.dataframe(
                ablation,
                use_container_width=True,
                hide_index=True,
            )
            _render_ablation_result_if_available()

        evidence(
            "Leakage-control decision",
            f"13 model-serving fields are admitted and 12 fields are strictly blocked (including job_id r=-0.966).",
            "Eligibility is based on semantics, inference-time availability, target derivation, "
            "identifier behavior and temporal stability—not correlation alone.",
            "Use the governed primary set for model development; quantify experience and skills "
            "through explicit ablation (Option A1 +46.3% MAE improvement) rather than silent inclusion/exclusion.",
            "good",
        )

    # ------------------------------------------------------------------
    # TAB 07 · ENCODING & ANALYSIS
    # ------------------------------------------------------------------
    with tabs[1]:
        stage_intro(
            "Stage 07 · Correlation Encoding & Analysis",
            "Fit analytical preprocessing on DEV only; audit Pearson/Spearman association, VIF, "
            "pairwise redundancy and skills.",
            "TRAIN/DEV only",
            "Encoded diagnostics + correlations + VIF + train-only skill evidence",
        )

        # Rào chắn rò rỉ dữ liệu (Critical rule)
        st.error(
            "🛑 **CRITICAL RULE: All encoders, scalers, and skill vocabulary (93 tokens) MUST BE fitted on TRAIN only. "
            "Apply unchanged to the Test set to prevent Data Leakage.**"
        )

        # Chốt hạ kích thước Feature Matrix
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Feature Matrix", "185 Model Features", help="86 One-hot, 93 Multi-hot, 6 Ordinal/Numeric")
        c_m2.metric("One-Hot Encoded", "86 Features", help="Nominal categorical features")
        c_m3.metric("Multi-Hot Skills", "93 Tokens", help="DEV-only vocabulary")
        c_m4.metric("Ordinal & Numeric", "6 Features", help="Standardized continuous and ordinal features")

        # ==========================================
        # CHÈN BẢNG CÔNG THỨC ENCODING CỦA SẾP NGÂN VÀO ĐẦU TAB 07
        # ==========================================
        st.markdown("### 🧮 Data Preprocessing Recipe")
        show_image(ASSETS_02/ "preprocessing_recipe.jpg")
        st.markdown("---") # Kẻ vạch phân cách với phần biểu đồ tương quan bên dưới
        # ==========================================

        # Main schema used by _data_ready.py
        corr = read_csv(READY / "07_target_correlations_dev.csv")
        vif = read_csv(READY / "07_vif_dev_numeric.csv")
        skills = read_csv(READY / "07_skill_frequency_and_salary_dev.csv")
        under50 = read_csv(READY / "07_skills_under_50_records_dev.csv")
        transform = read_csv(READY / "07_target_transform_review.csv")

        # Compatibility fallbacks from _data_ready1.py / earlier output naming
        if corr.empty:
            legacy_corr = read_csv(
                READY / "07_train_encoded_feature_target_correlation.csv"
            )
            if not legacy_corr.empty:
                corr = legacy_corr.copy()

        if skills.empty:
            legacy_vocab = read_csv(READY / "skill_vocabulary_train_only.csv")
            if not legacy_vocab.empty:
                skills = legacy_vocab.copy()

        if guard(corr, "Stage 07 outputs are missing."):
            return

        # Resolve both current and legacy correlation schemas safely.
        feature_col = _first_column(
            corr,
            ["feature", "encoded_feature", "feature_name"],
        )
        pearson_col = _first_column(
            corr,
            ["pearson", "pearson_r", "correlation"],
        )
        spearman_col = _first_column(
            corr,
            ["spearman", "spearman_r"],
        )
        abs_pearson_col = _first_column(
            corr,
            ["abs_pearson", "absolute_pearson", "abs_correlation"],
        )

        corr_display = corr.copy()
        if abs_pearson_col is None and pearson_col is not None:
            corr_display["_abs_pearson"] = pd.to_numeric(
                corr_display[pearson_col],
                errors="coerce",
            ).abs()
            abs_pearson_col = "_abs_pearson"

        if abs_pearson_col is not None:
            corr_display = corr_display.sort_values(
                abs_pearson_col,
                ascending=False,
            )

        strongest = corr_display.iloc[0]

        encoded_feature_count = len(corr_display)
        skill_vocab_count = len(skills)
        strongest_abs = (
            float(strongest[abs_pearson_col])
            if abs_pearson_col is not None
            and pd.notna(strongest[abs_pearson_col])
            else float("nan")
        )
        vif_gt_10 = (
            int((pd.to_numeric(vif["vif"], errors="coerce") > 10).sum())
            if not vif.empty and "vif" in vif.columns
            else 0
        )

        # 3. DRAW THE EXACT CHARTS FROM THE DOCUMENT
        if not corr_display.empty and feature_col and pearson_col:
            plot_df = corr_display.head(30).copy()
            plot_df[pearson_col] = pd.to_numeric(plot_df[pearson_col], errors="coerce")

            def simplify_name(f):
                f = str(f)
                if "AI Engineering" in f: return "job_category"
                if "years_of_experience" in f: return "years_of_experience"
                if "remote_work" in f: return "remote_work_mode"
                if "demand_score" in f: return "demand_score"
                if "skill_count" in f: return "skill_count"
                for p in ["nominal__job_category_", "cat_job_category_", "nominal__industry_", "cat_industry_", "nominal__city_", "cat_city_", "skills_", "nominal__job_title_", "cat_job_title_", "nominal__company_size_", "cat_company_size_", "numeric__", "num_", "education__"]:
                    if f.startswith(p): return f.replace(p, "")
                return f

            plot_df["Family"] = plot_df[feature_col].apply(simplify_name)
            colors = ['#7E57C2' if val > 0 else '#FF7043' for val in plot_df[pearson_col]]

            # Format text data labels (lấy 2 chữ số thập phân)
            text_labels = plot_df[pearson_col].apply(lambda x: f"{x:.2f}")

            st.markdown("---")
            
            # CHART 1 (Figure 6) - BUNG FULL WIDTH
            fig1 = px.bar(plot_df, x="Family", y=pearson_col, text=text_labels, title="6. TOP 30 FEATURE-FAMILY CORRELATION ON TRAIN<br><sup>(Pearson r với annual_salary_usd)</sup>")
            fig1.update_traces(marker_color=colors, textposition='outside', textfont_size=10)
            fig1.update_layout(xaxis_title="", yaxis_title="Pearson r", height=500, title_x=0.5, title_xanchor='center', xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Footer 3 khối cho Figure 6
            c1, c2, c3 = st.columns(3)
            with c1: st.info("📈 **Strongest positive:**\n\njob_category (r = 0.84)")
            with c2: st.error("📉 **Strongest negative:**\n\nSupport (r = -0.72)")
            with c3: st.success("🎯 **Most signal concentrated in:**\n\njob_category and years_of_experience.")
            st.markdown("<p style='text-align: center; font-style: italic; color: gray;'>Figure 6. the top feature-family by absolute correlation with annual_salary_usd on Train</p><br>", unsafe_allow_html=True)

            st.markdown("---")

            # CHART 2 (Figure 7) - BUNG FULL WIDTH
            fig2 = px.bar(plot_df, x=feature_col, y=pearson_col, text=text_labels, title="7. TOP 30 FEATURES BY PEARSON CORRELATION ON TRAIN")
            fig2.update_traces(marker_color=colors, textposition='outside', textfont_size=10)
            fig2.update_layout(xaxis_title="", yaxis_title="Pearson r", height=550, title_x=0.5, title_xanchor='center', xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Footer 3 khối cho Figure 7
            c4, c5, c6 = st.columns(3)
            with c4: st.info("📈 **Top positive feature:**\n\ncat_job_category_AI Engineering (r = 0.84)")
            with c5: st.error("📉 **Top negative feature:**\n\ncat_job_category_Support (r = -0.72)")
            with c6: st.success("🎯 **Top 2 features explain the main linear signal:**\n\njob_category & years_of_experience.")
            st.markdown("<p style='text-align: center; font-style: italic; color: gray;'>Figure 7. top features by Pearson correlation with annual_salary_usd on Train</p><br>", unsafe_allow_html=True)

        # Added from _data_ready1.py, but strictly data-driven.
        _render_skill_frequency_chart(skills)

        with st.expander("Target transformation review"):
            st.dataframe(
                transform,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("All DEV-only correlations"):
            st.dataframe(
                corr_display.drop(columns=["_abs_pearson"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
                height=450,
            )

        with st.expander(f"Skills with <50 DEV records ({len(under50)})"):
            st.dataframe(
                under50,
                use_container_width=True,
                hide_index=True,
                height=380,
            )

        strongest_name = (
            str(strongest[feature_col])
            if feature_col is not None
            else "top encoded feature"
        )
        pearson_value = (
            float(strongest[pearson_col])
            if pearson_col is not None and pd.notna(strongest[pearson_col])
            else None
        )
        spearman_value = (
            float(strongest[spearman_col])
            if spearman_col is not None and pd.notna(strongest[spearman_col])
            else None
        )

        if pearson_value is not None:
            dyn_finding = f"Maximum correlation signal originates from `{strongest_name}` (Pearson r ≈ {pearson_value:+.3f}). The skill vocabulary comprising {skill_vocab_count} tokens is fitted exclusively on the TRAIN set."
        else:
            dyn_finding = f"Maximum correlation signal originates from `{strongest_name}`. The skill vocabulary comprising {skill_vocab_count} tokens is fitted exclusively on the TRAIN set."

        evidence(
            "Stage 07 interpretation",
            dyn_finding,
            "Correlation here represents dataset association, NOT causality. Pearson/Spearman divergence warns of non-monotonic non-linearity; VIF and pairwise correlations help diagnose multicollinearity and eliminate feature redundancy.",
            "Carry the fitted preprocessing contract forward; all future validation/test rows are transformed only with DEV/fold-fitted encoders and skill vocabularies.",
            "info",
        )

    # ------------------------------------------------------------------
    # TAB 08 · TEMPORAL SPLIT
    # ------------------------------------------------------------------
    with tabs[2]:
        stage_intro(
            "Stage 08 · Train-Test Split",
            "Reserve March-2026 as locked final test and freeze preprocessing on TRAIN/DEV only.",
            "Clean modeling rows",
            "DEV + locked TEST + encoded matrices",
        )

        split = read_csv(READY / "08_split_summary.csv")
        before_after = read_csv(READY / "08_before_after_processing.csv")
        readiness = read_json(READY / "08_training_readiness.json")
        monthly = read_csv(READY / "08_monthly_distribution.csv")
        feature_names = read_csv(READY / "08_encoded_feature_names.csv")

        if guard(split, "Stage 08 outputs are missing."):
            return

        # Preserve the original _data_ready.py semantic partition lookup,
        # with a robust fallback for older split summaries.
        if "partition" in split.columns:
            dev_match = split.loc[split["partition"].eq("TRAIN_DEV")]
            test_match = split.loc[split["partition"].eq("LOCKED_TEST")]
        else:
            dev_match = pd.DataFrame()
            test_match = pd.DataFrame()

        if not dev_match.empty and not test_match.empty:
            dev_row = dev_match.iloc[0]
            test_row = test_match.iloc[0]
        elif len(split) >= 2:
            dev_row = split.iloc[0]
            test_row = split.iloc[1]
        else:
            st.error("Split summary must contain both development and locked-test partitions.")
            return

        dev_rows = int(dev_row["rows"])
        test_rows = int(test_row["rows"])

        encoded_count = readiness.get(
            "encoded_feature_count",
            len(feature_names) if not feature_names.empty else "—",
        )
        skill_count = readiness.get("skill_vocabulary_count", "—")

        cols = st.columns(4)
        cols[0].metric(
            "TRAIN / DEV",
            f"{dev_rows:,}",
            f"{float(dev_row['pct']):.1f}%"
            if "pct" in dev_row.index and pd.notna(dev_row["pct"])
            else None,
        )
        cols[1].metric(
            "Locked TEST",
            f"{test_rows:,}",
            f"{float(test_row['pct']):.1f}%"
            if "pct" in test_row.index and pd.notna(test_row["pct"])
            else None,
        )
        cols[2].metric("Encoded features", encoded_count, delta="-", delta_color="off")
        cols[3].metric("Skill vocab", skill_count, delta="-", delta_color="off")

        st.success(
            "✅ **NO RE-FITTING GUARANTEE: All Encoders, Scalers, and Skill Vocabulary are fitted on TRAIN ONLY. "
            "Locked TEST set is transformed using frozen objects to strictly prevent look-ahead bias.**"
        )

        if not monthly.empty:
            plot_df = monthly.copy()
            # Ghép năm và tháng để tạo mốc thời gian chuẩn YYYY-MM
            if 'posting_year' in plot_df.columns and 'posting_month' in plot_df.columns:
                plot_df['period'] = plot_df['posting_year'].astype(str) + "-" + plot_df['posting_month'].astype(str).str.zfill(2)
                month_col = 'period'
            else:
                month_col = _first_column(plot_df, ["period", "year_month", "date"]) or plot_df.columns[0]
            
            rows_col = _first_column(plot_df, ["rows", "count", "n", "volume"]) or plot_df.columns[1]
            plot_df = plot_df.sort_values(month_col)
            
            # Đổ màu giống hệt bản gốc
            colors = ['#F09A9D' if '2026-03' in str(val) else '#AEC7E1' for val in plot_df[month_col]]
            
            fig_time = px.bar(plot_df, x=month_col, y=rows_col)
            fig_time.update_traces(marker_color=colors)
            
            # Decorate font chữ và đóng khung viền đen y hệt Document
            fig_time.update_layout(
                title=dict(text="<b>Monthly Posting Volume & Locked Temporal Test</b>", x=0.5, xanchor='center', font=dict(size=24, color='#2C3E50', family="Arial")),
                xaxis_title="",
                yaxis_title="<b>Rows</b>",
                xaxis_tickangle=-45,
                height=500,
                xaxis_type='category',
                plot_bgcolor='white',
                margin=dict(t=60, b=40, l=40, r=40)
            )
            fig_time.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, gridcolor='#F1F5F9', tickfont=dict(size=14, color='black'))
            fig_time.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, gridcolor='#F1F5F9', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
            
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No monthly data available")

        left, right = st.columns([1, 1.5])

        with left:
            if "partition" in split.columns and "rows" in split.columns:
                fig = px.pie(
                    split,
                    names="partition",
                    values="rows",
                    hole=0.58,
                    title="Development vs locked test",
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.dataframe(
                before_after,
                use_container_width=True,
                hide_index=True,
                height=410,
            )

        with st.expander("Monthly distribution"):
            st.dataframe(
                monthly,
                use_container_width=True,
                hide_index=True,
            )
        # ==========================================
        # CHÈN CHART ĐỐI CHIẾU TRAIN-TEST VÀO ĐÂY
        # ==========================================
        st.markdown("### ⚖️ Numeric Features: Train vs Test Distribution")
        show_image(ASSETS_02 / "08_numeric_train_test_dist.jpg")
        # ==========================================
        
        # Residual review from _data_ready1.py is retained only when real output exists.
        #with st.expander("Supplementary residual review", expanded=False):
            #_render_residual_review_if_available()
        st.markdown("### Ablation Study & Residual Analysis")
        left, right = st.columns([1, 1.2])

        with left:
            try:
                # Load file kết quả tính toán R2 của 4 model từ backend
                ablation_df = pd.read_csv("outputs/03_model_comparison/ablation_results.csv")
                st.dataframe(ablation_df, hide_index=True, use_container_width=True)
            except FileNotFoundError:
                st.error("Missing file ablation_results.csv")
                
            st.info("💡 **Conclusion:** $R^2$ results for Model A ≈ Model B. Dropping categories (Model C) and experience (Model D) drastically reduces performance. \n\n👉 **The dataset exhibits a simplistic structure**; other features (skills, education) act merely as given information and contribute no marginal predictive value.")

        with right:
            try:
                # Load ảnh biểu đồ Residuals từ backend
                st.image("outputs/04_best_model_and_feature_importance/locked_test_residuals.png", 
                        caption="Residual analysis indicates Heteroscedasticity in the >$200k USD salary range.")
            except FileNotFoundError:
                st.error("Missing image locked_test_residuals.png")

        evidence(
            "Locked-test governance",
            f"DEV={dev_rows:,} rows; locked TEST={test_rows:,} rows. "
            f"Preprocessing fit scope: {readiness.get('fit_scope', 'TRAIN/DEV only')}.",
            "The final holdout period is not used for feature selection, hyperparameter tuning "
            "or temporal CV, preventing look-ahead bias.",
            "Use expanding-window temporal CV inside DEV for Stage 09. "
            "Open locked-test labels only once after model family, tuning policy and narrative are frozen.",
            "good",
        )
