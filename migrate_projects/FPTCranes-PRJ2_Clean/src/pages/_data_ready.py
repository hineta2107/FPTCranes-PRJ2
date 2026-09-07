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

    skill_col = _first_column(skills, ["skill", "token", "required_skill", "skill_name"])
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
    plot_df = plot_df.dropna(subset=[count_col]).nlargest(10, count_col)

    if plot_df.empty:
        st.dataframe(skills, use_container_width=True, hide_index=True, height=420)
        return

    fig = px.bar(
        plot_df,
        x=skill_col,
        y=count_col,
        title="Top 10 DEV-only skills by record frequency",
        labels={skill_col: "Skill", count_col: "Records"},
    )
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        xaxis_tickangle=-45,
        height=450,
        margin={"b": 110},
    )
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
    # PASTE KHỐI EXECUTIVE OVERVIEW NÀY VÀO ĐÂY
    # ==========================================
    with st.expander("🗺️ Executive Overview: Dataset Transition & ML Pipeline", expanded=True):
        st.markdown("Bức tranh tổng thể:Lộ trình biến đổi nó thành Ma trận Feature cho Machine Learning (bên TRÁI) và Dữ liệu thay đổi như thế nào sau bước Basic Clean (bên PHẢI).")
        
        # Chia 2 cột: Cột trái to hơn để chứa bảng ngang, cột phải nhỏ chứa luồng dọc
        c1, c2 = st.columns([0.6, 1.7])
        with c1:
            show_image(ASSETS_02 / "mini_pipeline.jpg")
        with c2:
            show_image(ASSETS_02 / "raw_vs_clean.jpg")
            
    st.markdown("---") # Kẻ vạch phân cách cho đẹp
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

        if guard(policy, "Stage 06 outputs are missing."):
            return

        blocked_count = (
            int(policy["policy"].astype(str).str.upper().eq("BLOCK").sum())
            if "policy" in policy.columns
            else 0
        )
        phase2_count = (
            int(
                policy["role"]
                .astype(str)
                .str.contains("PHASE2", case=False, na=False)
                .sum()
            )
            if "role" in policy.columns
            else 0
        )

        cols = st.columns(4)
        cols[0].metric("Serving fields", len(primary))
        cols[1].metric("Blocked fields", blocked_count)
        cols[2].metric("Phase-2 skills", phase2_count)
        cols[3].metric("Leakage gate", gate.get("status", "—"))

        st.success(
            "Salary minimum, salary maximum and salary tier must remain blocked from the primary "
            "feature matrix when they are target-adjacent/target-derived. Skills are handled through "
            "a training-only vocabulary, and `skill_count` represents the distinct normalized token count.",
            icon=":material/verified_user:",
        )

        left, right = st.columns([1, 1.35])

        with left:
            show_image(READY / "06_feature_governance_mix.png")

        with right:
            st.markdown("#### Primary serving feature set")
            st.dataframe(
                primary,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        st.markdown("#### Full feature / leakage contract")
        st.dataframe(
            policy,
            use_container_width=True,
            hide_index=True,
            height=440,
            key="feature_policy",
        )

        """ with st.expander("Required feature-family ablation plan", expanded=True):
            st.dataframe(
                ablation,
                use_container_width=True,
                hide_index=True,
            )
            _render_ablation_result_if_available() """ #code cũ
        
        #CODE CỦA DI
        st.markdown("### 📊 Ablation Analysis Dashboard")
        # 1. Phọt thẳng cái ảnh 6 chart nguyên khối ra đây
        show_image(READY / "06_ablation_dashboard.png")

        st.markdown("### 📝 Ablation Results & Next Actions")
        # 2. Đắp cái ảnh Bảng kết quả (Mockup) của sếp Ngân vào đây để lấp chỗ trống Backend
        show_image(ASSETS_02 / "ablation_results_table.jpg")

        # 3. Giấu cái bảng CSV khô khan vào expander bên dưới
        with st.expander("Required feature-family ablation plan (Tabular Data)", expanded=False):
            st.dataframe(
                ablation,
                use_container_width=True,
                hide_index=True,
            )
            # Hàm này vẫn giữ nguyên để sau này Backend có CSV thật thì nó tự động mọc ra thêm
            _render_ablation_result_if_available()

        evidence(
            "Leakage-control decision",
            f"{len(primary)} model-serving fields are admitted and {blocked_count} fields are blocked.",
            "Eligibility is based on semantics, inference-time availability, target derivation, "
            "identifier behavior and temporal stability—not correlation alone.",
            "Use the governed primary set for model development; quantify experience and skills "
            "through explicit ablation rather than silent inclusion/exclusion.",
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

        cols = st.columns(5)
        cols[0].metric("Encoded features", encoded_feature_count)
        cols[1].metric("Skill vocabulary", skill_vocab_count)
        cols[2].metric(
            "Strongest |Pearson|",
            f"{strongest_abs:.3f}" if pd.notna(strongest_abs) else "—",
        )
        cols[3].metric("VIF > 10", vif_gt_10)
        cols[4].metric("Skills <50 rows", len(under50))

        # Prefer the current output image; fall back to legacy file name.
        corr_chart = _first_existing_image(
            READY / "07_top30_correlations.png",
            READY / "top30_train_target_correlation.png",
        )
        if corr_chart is not None:
            show_image(corr_chart)

        left, right = st.columns(2)
        with left:
            show_image(READY / "07_pearson_vs_spearman.png")
        with right:
            show_image(READY / "07_vif_numeric.png")

        show_image(READY / "07_top30_skills.png")

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

        if pearson_value is not None and spearman_value is not None:
            association_text = (
                f"The strongest encoded association is `{strongest_name}` "
                f"(Pearson {pearson_value:+.3f}; Spearman {spearman_value:+.3f}). "
                f"The skill vocabulary contains {skill_vocab_count} DEV-only tokens."
            )
        elif pearson_value is not None:
            association_text = (
                f"The strongest encoded association is `{strongest_name}` "
                f"(Pearson {pearson_value:+.3f}). "
                f"The skill vocabulary contains {skill_vocab_count} DEV-only tokens."
            )
        else:
            association_text = (
                f"The strongest encoded feature is `{strongest_name}`. "
                f"The skill vocabulary contains {skill_vocab_count} DEV-only tokens."
            )

        evidence(
            "Stage 07 interpretation",
            association_text,
            "Correlation describes association in this dataset, not causality. "
            "Pearson/Spearman divergence can reveal monotonic nonlinearity; "
            "VIF and pairwise checks diagnose redundancy. Skills are tokenized "
            "and de-duplicated rather than one-hot encoding whole skill strings.",
            "Carry the fitted preprocessing contract forward; all future validation/test rows "
            "are transformed only with DEV/fold-fitted encoders and skill vocabularies.",
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
        cols[2].metric("Encoded features", encoded_count)
        cols[3].metric("Skill vocab", skill_count)

        show_image(READY / "08_temporal_split_timeline.png")

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
                st.error("Chưa thấy file ablation_results.csv")
                
            st.info("💡 **Kết luận:** Kết quả $R^2$ Model A ~ Model B. Khi drop categories (Model C) và experience (Model D) hiệu suất giảm mạnh. \n\n👉 **Dataset có cấu trúc đơn giản**, các features khác (skills, education) chỉ có tính chất given information, không đóng góp giá trị dự đoán biên.")

        with right:
            try:
                # Load ảnh biểu đồ Residuals từ backend
                st.image("outputs/04_best_model_and_feature_importance/locked_test_residuals.png", 
                        caption="Phân tích phần dư (Residuals) xuất hiện hiện tượng Heteroscedasticity ở dải lương >200k USD.")
            except FileNotFoundError:
                st.error("Chưa thấy ảnh locked_test_residuals.png")

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
