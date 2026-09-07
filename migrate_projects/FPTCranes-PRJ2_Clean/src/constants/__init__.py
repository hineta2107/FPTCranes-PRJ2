from __future__ import annotations

TARGET = "annual_salary_usd"
SEED = 42
LOCKED_YEAR = 2026
LOCKED_MONTH = 3

STAGES = [
    "Load Data",
    "Project Scope & Initial Inspection",
    "Data Quality Check",
    "Corrupted Row Removal",
    "Contradictory-Feature Investigation",
    "Feature Selection & Leakage Prevention",
    "Correlation Encoding & Analysis",
    "Train-Test Split",
    "Model Training & Comparison",
    "Best Model Selection & Feature Importance Review",
    "Save Deployable Pipeline + Metadata",
    "Streamlit Salary Prediction Dashboard",
]

STRUCTURED_FEATURES = [
    "job_title",
    "job_category",
    "years_of_experience",
    "education_required",
    "city",
    "country",
    "remote_work",
    "company_size",
    "industry",
    "demand_score",
    "benefits_score_10",
]
MODEL_FEATURES = STRUCTURED_FEATURES + ["required_skills", "skill_count"]
NOMINAL_FEATURES = [
    "job_title",
    "job_category",
    "city",
    "country",
    "remote_work",
    "industry",
]
ORDINAL_FEATURES = ["education_required", "company_size"]
NUMERIC_FEATURES = ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
EDUCATION_ORDER = ["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"]
COMPANY_SIZE_ORDER = [
    "Startup (1-50)",
    "SME (51-500)",
    "Mid-size (501-5000)",
    "Enterprise (5000+)",
    "Big Tech (FAANG+)",
]
BLOCKED_FEATURES = [
    "job_id",
    "salary_min_usd",
    "salary_max_usd",
    "salary_tier",
    "experience_level",
    "posting_year",
    "posting_month",
    "is_senior",
    "is_remote_friendly",
    "is_llm_role",
    "ai_salary_premium_pct",
    "demand_growth_yoy_pct",
]

PASTEL = {
    "blue": "#BFD7EA",
    "blue_dark": "#6C9BCF",
    "purple": "#D9CCEB",
    "pink": "#F6C6D8",
    "orange": "#F6C89F",
    "yellow": "#F4E3A1",
    "green": "#CDE6C7",
    "ink": "#25304A",
    "muted": "#64748B",
    "border": "#E4DFF0",
}
