from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, guard, read_csv, show_image

OUT = Config.OUTPUT_DIR / "03_model_comparison"


def render() -> None:
    page_header("3. Model comparison", "Stage 09 · Dummy performance floor + five regression candidates on identical expanding-window temporal CV", "🏁")
    stage_intro("Stage 09 · Model Training & Comparison", "Select the winning model family using development-period evidence only.", "TRAIN/DEV + 5 expanding temporal folds", "Model leaderboard + fold stability + ablation + importance drift + runtime performance")
    comp = read_csv(OUT / "09_model_comparison_temporal_cv.csv")
    folds = read_csv(OUT / "09_model_comparison_fold_metrics.csv")
    ablation = read_csv(OUT / "09_feature_family_ablation.csv")
    drift = read_csv(OUT / "09_feature_importance_by_fold.csv")
    runtime = read_csv(OUT / "09_model_runtime_performance.csv")

    if guard(comp, "Stage 09 outputs are missing. Run `python pipeline.py`."):
        return

    candidate = comp.loc[~comp["is_baseline"].astype(bool)].sort_values("CV_MAE_mean").iloc[0]
    dummy = comp.loc[comp["is_baseline"].astype(bool)]
    dummy_mae = float(dummy.iloc[0]["CV_MAE_mean"]) if not dummy.empty else float("nan")

    # ── Summary metrics ───────────────────────────────────────────────────────
    cols = st.columns(5)
    cols[0].metric("Selected family", candidate["model"])
    cols[1].metric("CV MAE", fmt_money(candidate["CV_MAE_mean"]))
    cols[2].metric("CV RMSE", fmt_money(candidate["CV_RMSE_mean"]))
    cols[3].metric("CV R²", f"{float(candidate['CV_R2_mean']):.3f}")
    cols[4].metric("MAE vs Dummy", f"{(1-float(candidate['CV_MAE_mean'])/dummy_mae)*100:.1f}% better" if dummy_mae == dummy_mae else "—")

    # ── Leaderboard table ─────────────────────────────────────────────────────
    table = comp.copy()
    round_cols = ["CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_RMSE_std", "CV_MedAE_mean"]
    for c in round_cols:
        if c in table:
            table[c] = table[c].round(1)
    if "fit_seconds_mean" in table:
        table["fit_seconds_mean"] = table["fit_seconds_mean"].round(4)
    if "predict_seconds_mean" in table:
        table["predict_seconds_mean"] = table["predict_seconds_mean"].round(4)
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ── CV charts ─────────────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        show_image(OUT / "model_comparison_cv_mae.png")
    with right:
        show_image(OUT / "model_comparison_cv_r2.png")
    show_image(OUT / "09_fold_stability_mae.png")

    if not folds.empty:
        with st.expander("Fold-by-fold metrics"):
            st.dataframe(folds, use_container_width=True, hide_index=True, height=420)

    # ── 🕐 Runtime Performance ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🕐 Runtime Performance")
    st.caption(
        "Thời gian đo bằng `time.perf_counter()` bao gồm cả preprocessing. "
        "Đơn vị: giây / fold trung bình. "
        "SVR đã bị loại bỏ do hiệu suất kém (CV MAE ≈ Dummy Baseline)."
    )

    show_image(OUT / "model_comparison_training_time.png")

    if not runtime.empty:
        rt_display = runtime.copy()
        for col in ["fit_seconds_mean", "predict_seconds_mean"]:
            if col in rt_display.columns:
                rt_display[col] = rt_display[col].apply(lambda x: f"{x:.4f}s")
        if "fit_seconds_mean" in runtime.columns and "predict_seconds_mean" in runtime.columns:
            rt_display["total_avg_sec"] = (
                runtime["fit_seconds_mean"] + runtime["predict_seconds_mean"]
            ).apply(lambda x: f"{x:.4f}s")
        st.dataframe(rt_display, use_container_width=True, hide_index=True)

        # Plotly bar chart thời gian chạy
        if "fit_seconds_mean" in runtime.columns and not runtime.empty:
            fig_rt = px.bar(
                runtime.sort_values("fit_seconds_mean"),
                x="fit_seconds_mean",
                y="model",
                orientation="h",
                title="Average Fit Time per Fold (seconds)",
                labels={"fit_seconds_mean": "Avg fit time (s)", "model": "Model"},
                color="fit_seconds_mean",
                color_continuous_scale="Blues",
            )
            fig_rt.update_layout(coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.info("Runtime CSV not found. Run `python pipeline.py` to generate `09_model_runtime_performance.csv`.")

    # ── Feature-family ablation ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Feature-family ablation")
    show_image(OUT / "09_feature_family_ablation.png")
    st.dataframe(ablation, use_container_width=True, hide_index=True)

    # ── Temporal importance stability ─────────────────────────────────────────
    st.markdown("#### Temporal importance stability / concept-drift diagnostic")
    show_image(OUT / "09_feature_importance_drift.png")
    with st.expander("Importance by fold"):
        st.dataframe(drift, use_container_width=True, hide_index=True, height=400)

    evidence(
        "Model-family selection",
        f"`{candidate['model']}` has the lowest mean temporal-CV MAE among non-baseline candidates: {fmt_money(candidate['CV_MAE_mean'])}.",
        "The winner is chosen from future-facing development folds; the March-2026 locked test is not used to rank models. Fold variance, ablation and importance drift qualify the score.",
        "Freeze the winning family and bounded tuning policy, then proceed to one-time locked-test evaluation in Stage 10.",
        "good",
    )
