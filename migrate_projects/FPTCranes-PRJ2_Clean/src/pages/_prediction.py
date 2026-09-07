from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, read_csv, read_json, show_image

ART = Config.ARTIFACT_DIR
OUT = Config.OUTPUT_DIR / "05_salary_prediction"


@st.cache_resource
def _load_bundle():
    path = ART / "model_bundle.joblib"
    return joblib.load(path) if path.exists() else None


def render() -> None:
    page_header("5. Salary prediction", "Stage 12 · authenticated real-time inference using the exact saved preprocessing + model bundle", "💰")
    stage_intro("Stage 12 · Streamlit Salary Prediction Dashboard", "Build a valid raw feature row from training metadata, predict with the same bundle, and attach interval/OOD review context.", "User form + artifacts/metadata.json", "Predicted salary + empirical 90% error interval + review flags")
    metadata = read_json(ART / "metadata.json")
    bundle = _load_bundle()
    if not metadata or bundle is None:
        st.error("Deployment artifacts are missing. Run `python pipeline.py` first.")
        return
    cats = metadata.get("category_options", {})
    ranges = metadata.get("numeric_ranges", {})
    skills = metadata.get("skill_vocabulary", [])
    interval = float(metadata.get("prediction_interval_abs_error_q90", 0.0))

    with st.form("salary_prediction_form"):
        a, b, c = st.columns(3)
        job_title = a.selectbox("Job title", cats.get("job_title", ["Unknown"]))
        job_category = b.selectbox("Job category", cats.get("job_category", ["Unknown"]))
        education = c.selectbox("Education", cats.get("education_required", ["Unknown"]))
        city = a.selectbox("City", cats.get("city", ["Unknown"]))
        country = b.selectbox("Country", cats.get("country", ["Unknown"]))
        remote = c.selectbox("Remote work", cats.get("remote_work", ["Unknown"]))
        company = a.selectbox("Company size", cats.get("company_size", ["Unknown"]))
        industry = b.selectbox("Industry", cats.get("industry", ["Unknown"]))
        yr = ranges.get("years_of_experience", {"min": 1, "max": 15, "median": 6})
        demand_r = ranges.get("demand_score", {"min": 0, "max": 100, "median": 80})
        benefit_r = ranges.get("benefits_score_10", {"min": 0, "max": 10, "median": 8})
        years = c.number_input("Years of experience", min_value=float(yr.get("min", 0)), max_value=float(yr.get("max", 50)), value=float(yr.get("median", 5)), step=1.0)
        demand = a.number_input("Demand score", min_value=float(demand_r.get("min", 0)), max_value=float(demand_r.get("max", 100)), value=float(demand_r.get("median", 80)), step=1.0)
        benefits = b.number_input("Benefits score / 10", min_value=float(benefit_r.get("min", 0)), max_value=float(benefit_r.get("max", 10)), value=float(benefit_r.get("median", 8)), step=1.0)
        selected_skills = st.multiselect("Required skills", skills, max_selections=12)
        submitted = st.form_submit_button("🚀 Predict annual salary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "job_title": job_title,
            "job_category": job_category,
            "years_of_experience": years,
            "education_required": education,
            "city": city,
            "country": country,
            "remote_work": remote,
            "company_size": company,
            "industry": industry,
            "demand_score": demand,
            "benefits_score_10": benefits,
            "required_skills": "|".join(dict.fromkeys(selected_skills)),
            "skill_count": len(set(selected_skills)),
        }])
        model_features = metadata["model_features"]
        prediction = float(bundle.predict(row[model_features])[0])
        low, high = max(0.0, prediction - interval), prediction + interval
        cols = st.columns(3)
        cols[0].metric("Predicted salary", fmt_money(prediction))
        cols[1].metric("Practical lower bound", fmt_money(low))
        cols[2].metric("Practical upper bound", fmt_money(high))
        review = []
        if not selected_skills:
            review.append("No skill tokens selected; skill_count=0.")
        st.success(f"Model: **{metadata.get('model_name','—')}**. Empirical interval half-width: **{fmt_money(interval)}**.")
        if review:
            st.warning("Review flag: " + " ".join(review))
        evidence("Prediction interpretation", f"Point estimate {fmt_money(prediction)} with empirical ±{fmt_money(interval)} 90th-percentile DEV out-of-fold absolute-error band.", "This is a practical error band derived from historical validation residuals, not a formal probabilistic confidence guarantee and not a causal market valuation.", "Use for academic scenario exploration; obtain external verified market data before operational compensation decisions.", "info")

    summary = read_csv(OUT / "12_prediction_summary.csv")
    if not summary.empty:
        st.markdown("#### Final locked-test serving summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        show_image(OUT / "salary_prediction_actual_vs_predicted.png")
    with right:
        show_image(OUT / "salary_prediction_residuals.png")
    with st.expander("Locked-test prediction examples"):
        st.dataframe(read_csv(OUT / "12_locked_test_prediction_examples.csv").head(40), use_container_width=True, hide_index=True)
    st.caption("Geographic cost-of-living adjustment is intentionally not implemented because the supplied project data does not include a verified cost-of-living index.")
