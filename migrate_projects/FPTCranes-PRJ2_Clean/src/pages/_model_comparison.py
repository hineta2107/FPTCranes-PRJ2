from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, guard, read_csv, show_image

OUT = Config.OUTPUT_DIR / "03_model_comparison"


def render() -> None:
    page_header("3. Model comparison", "Stage 09 · Dummy performance floor + five regression candidates on identical expanding-window temporal CV", "🏁")
    stage_intro("Stage 09 · Model Training & Comparison", "Select the winning model family using development-period evidence only.", "TRAIN/DEV + 5 expanding temporal folds", "Model leaderboard + fold stability + ablation + importance drift")
    comp = read_csv(OUT / "09_model_comparison_temporal_cv.csv")
    folds = read_csv(OUT / "09_model_comparison_fold_metrics.csv")
    ablation = read_csv(OUT / "09_feature_family_ablation.csv")
    drift = read_csv(OUT / "09_feature_importance_by_fold.csv")
    if guard(comp, "Stage 09 outputs are missing. Run `python pipeline.py`."):
        return
    candidate = comp.loc[~comp["is_baseline"].astype(bool)].sort_values("CV_MAE_mean").iloc[0]
    dummy = comp.loc[comp["is_baseline"].astype(bool)]
    dummy_mae = float(dummy.iloc[0]["CV_MAE_mean"]) if not dummy.empty else float("nan")
    cols = st.columns(5)
    cols[0].metric("Selected family", candidate["model"])
    cols[1].metric("CV MAE", fmt_money(candidate["CV_MAE_mean"]))
    cols[2].metric("CV RMSE", fmt_money(candidate["CV_RMSE_mean"]))
    cols[3].metric("CV R²", f"{float(candidate['CV_R2_mean']):.3f}")
    cols[4].metric("MAE vs Dummy", f"{(1-float(candidate['CV_MAE_mean'])/dummy_mae)*100:.1f}% better" if dummy_mae == dummy_mae else "—")
    table = comp.copy()
    for c in ["CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_RMSE_std", "CV_MedAE_mean"]:
        if c in table:
            table[c] = table[c].round(1)
    st.dataframe(table, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        show_image(OUT / "model_comparison_cv_mae.png")
    with right:
        show_image(OUT / "model_comparison_cv_r2.png")
    show_image(OUT / "09_fold_stability_mae.png")
    if not folds.empty:
        with st.expander("Fold-by-fold metrics"):
            st.dataframe(folds, use_container_width=True, hide_index=True, height=420)
    st.markdown("#### Feature-family ablation")
    show_image(OUT / "09_feature_family_ablation.png")
    st.dataframe(ablation, use_container_width=True, hide_index=True)
    st.markdown("#### Temporal importance stability / concept-drift diagnostic")
    show_image(OUT / "09_feature_importance_drift.png")
    with st.expander("Importance by fold"):
        st.dataframe(drift, use_container_width=True, hide_index=True, height=400)
    evidence("Model-family selection", f"`{candidate['model']}` has the lowest mean temporal-CV MAE among non-baseline candidates: {fmt_money(candidate['CV_MAE_mean'])}.", "The winner is chosen from future-facing development folds; the March-2026 locked test is not used to rank models. Fold variance, ablation and importance drift qualify the score.", "Freeze the winning family and bounded tuning policy, then proceed to one-time locked-test evaluation in Stage 10.", "good")
