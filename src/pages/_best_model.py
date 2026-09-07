from __future__ import annotations

import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, guard, read_csv, read_json, show_image

BEST = Config.OUTPUT_DIR / "04_best_model_and_feature_importance"
ART = Config.ARTIFACT_DIR
OUT_PRED = Config.OUTPUT_DIR / "05_salary_prediction"


def render() -> None:
    page_header("4. Best model", "Stage 10 · bounded tuning → one-time locked test → explainability → deployment-equivalence gate", "🏆")

    # ── Stage 10: Best Model Selection & Feature Importance ──────────────────
    stage_intro(
        "Stage 10 · Best Model Selection & Feature Importance Review",
        "Tune inside DEV folds, then open March-2026 once and inspect model behavior rather than accepting R² at face value.",
        "Frozen model family + DEV folds",
        "Locked-test metrics + predictions + residuals + importance + slices",
    )
    metrics = read_csv(BEST / "10_final_locked_test_metrics.csv")
    tuning = read_csv(BEST / "10_best_model_tuning_results.csv")
    raw_imp = read_csv(BEST / "10_raw_feature_permutation_importance.csv")
    enc_imp = read_csv(BEST / "10_encoded_feature_importance.csv")
    slices = read_csv(BEST / "10_error_slices.csv")

    if guard(metrics, "Stage 10 outputs are missing. Run `python pipeline.py` first."):
        return

    m = metrics.iloc[0]
    cols = st.columns(5)
    cols[0].metric("Best model", m["selected_model"])
    cols[1].metric("MAE", fmt_money(m["MAE"]))
    cols[2].metric("RMSE", fmt_money(m["RMSE"]))
    cols[3].metric("R²", f"{float(m['R2']):.3f}")
    cols[4].metric("MedAE", fmt_money(m["MedAE"]))

    left, right = st.columns(2)
    with left:
        show_image(BEST / "actual_vs_predicted_locked_test.png")
    with right:
        show_image(BEST / "locked_test_residuals.png")
    left, right = st.columns(2)
    with left:
        show_image(BEST / "raw_feature_permutation_importance.png")
    with right:
        show_image(BEST / "top25_encoded_feature_importance.png")

    with st.expander("Bounded tuning results (4 candidates)", expanded=True):
        st.dataframe(tuning, use_container_width=True, hide_index=True)
    with st.expander("Raw-feature permutation importance"):
        st.dataframe(raw_imp, use_container_width=True, hide_index=True)
    with st.expander("Encoded feature importance"):
        st.dataframe(enc_imp.head(50), use_container_width=True, hide_index=True)
    with st.expander("Error slices"):
        st.dataframe(slices, use_container_width=True, hide_index=True)

    # ── Final locked-test serving summary (chuyển từ trang 5) ────────────────
    summary = read_csv(OUT_PRED / "12_prediction_summary.csv")
    if not summary.empty:
        st.markdown("#### Final locked-test serving summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Locked-test prediction examples (chuyển từ trang 5) ─────────────────
    with st.expander("Locked-test prediction examples"):
        st.dataframe(
            read_csv(OUT_PRED / "12_locked_test_prediction_examples.csv").head(40),
            use_container_width=True,
            hide_index=True,
        )


    top_share = float(enc_imp.head(2)["importance"].sum()) if not enc_imp.empty and "importance" in enc_imp else 0.0
    evidence(
        "Locked-test interpretation",
        f"{m['selected_model']} achieves MAE {fmt_money(m['MAE'])}, RMSE {fmt_money(m['RMSE'])}, R² {float(m['R2']):.3f}, MedAE {fmt_money(m['MedAE'])}. Top-two encoded importance share is {top_share*100:.1f}%.",
        "These metrics quantify fit to this supplied dataset and future month; feature importance explains the fitted model, not causal salary economics. Concentrated importance warrants extra caution when underlying fields show synthetic-looking structure.",
        "Promote only with the documented limitations, ablation evidence and reload-equivalence PASS. External verified job-posting validation is still required for operational compensation decisions.",
        "warn",
    )

