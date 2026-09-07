from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import platform
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.builder.path_builder import PipelinePaths, build_paths
from src.config import Config
from src.constants import (
    BLOCKED_FEATURES,
    EDUCATION_ORDER,
    COMPANY_SIZE_ORDER,
    LOCKED_MONTH,
    LOCKED_YEAR,
    MODEL_FEATURES,
    SEED,
    STAGES,
    STRUCTURED_FEATURES,
    TARGET,
)
from src.training.build_preprocessor import build_preprocessor
from src.training.model_catalog import model_catalog
from src.training.temporal_cv import monthly_temporal_folds, regression_metrics
from src.training.tuning import tune_gradient_boosting, tune_random_forest
from src.utils.artifacts import save_csv, save_json, sha256_file
from src.utils.pipeline_plots import (
    plot_experience_salary_dashboard,
    plot_ablation,
    plot_best_outputs,
    plot_cardinality,
    plot_correlations,
    plot_feature_policy,
    plot_fold_stability,
    plot_importance_drift,
    plot_logic_issue_rates,
    plot_missingness,
    plot_model_comparison,
    plot_pearson_spearman,
    plot_pipeline_12,
    plot_consistency_audit_dashboard,
    plot_salary_by_job_category,
    plot_planned_ablation_dashboard,
    #plot_salary_range_integrity,
    plot_skills,
    plot_target_distribution,
    plot_temporal_split,
    plot_vif,

)


def stage_header(n: int, title: str) -> None:
    print(f"\n{'=' * 86}\nSTAGE {n:02d} — {title}\n{'=' * 86}")


def normalize_skill_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for token in str(value).split("|"):
        token = " ".join(token.strip().split())
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return "|".join(out)


def skill_count(value: Any) -> int:
    text = normalize_skill_string(value)
    return 0 if not text else len(text.split("|"))


def experience_bucket(years: Any) -> str:
    value = pd.to_numeric(pd.Series([years]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    if value <= 2:
        return "Entry (0-2 yrs)"
    if value <= 5:
        return "Mid (3-5 yrs)"
    if value <= 9:
        return "Senior (6-9 yrs)"
    return "Lead (10+ yrs)"


def salary_tier_expected(salary: float) -> str:
    if salary < 100_000:
        return "Entry (<$100k)"
    if salary < 150_000:
        return "Mid ($100-150k)"
    if salary < 200_000:
        return "Upper-Mid ($150-200k)"
    if salary <= 300_000:
        return "Senior ($200-300k)"
    return "Elite (>$300k)"


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "unique_pct": float(series.nunique(dropna=True) / len(df) * 100) if len(df) else 0.0,
                "sample_values": " | ".join(map(str, series.dropna().astype(str).unique()[:5])),
            }
        )
    return pd.DataFrame(rows)


def target_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (pd.to_numeric(df[TARGET], errors="coerce") < lower) | (pd.to_numeric(df[TARGET], errors="coerce") > upper)
    profile = pd.DataFrame(
        [
            {
                "count": int(target.size),
                "min": float(target.min()),
                "max": float(target.max()),
                "mean": float(target.mean()),
                "median": float(target.median()),
                "std": float(target.std()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_iqr_bound": float(lower),
                "upper_iqr_bound": float(upper),
                "outlier_count": int(mask.sum()),
                "high_outlier_count": int((pd.to_numeric(df[TARGET], errors="coerce") > upper).sum()),
            }
        ]
    )
    columns = [c for c in ["job_id", "job_title", "AI Engineering", "experience_level", "years_of_experience", TARGET, "posting_year", "posting_month"] if c in df.columns]
    outliers = df.loc[mask, columns].sort_values(TARGET, ascending=False)
    return profile, outliers


def categorical_value_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.select_dtypes(include=["object", "string", "category"]).columns:
        counts = df[column].fillna("<MISSING>").astype(str).str.strip().value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "pct": float(count / len(df) * 100),
                    "status": "INVALID_HEADER_TOKEN" if column == "AI Engineering" and value == "job_category" else "OBSERVED",
                }
            )
    return pd.DataFrame(rows)


def feature_policy_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        if feature == "required_skills":
            role = "PHASE2_MULTI_HOT"
        elif feature == "skill_count":
            role = "ENGINEERED"
        elif feature == "years_of_experience":
            role = "KEEP_WITH_ABLATION_WARNING"
        else:
            role = "KEEP"
        rows.append(
            {
                "feature": feature,
                "policy": "ALLOW",
                "role": role,
                "reason": "Available before target; transformed only inside TRAIN-fitted preprocessing.",
            }
        )
    for feature in BLOCKED_FEATURES:
        if feature in {"salary_min_usd", "salary_max_usd", "salary_tier"}:
            reason = "Target leakage / target-adjacent compensation metadata."
        elif feature == "experience_level":
            reason = "Contradictory/redundant with years_of_experience; primary model uses ablation evidence instead."
        else:
            reason = "Identifier, derived flag, temporal metadata or unverified inference-time/provenance field."
        rows.append({"feature": feature, "policy": "BLOCK", "role": "EXCLUDED", "reason": reason})
    return pd.DataFrame(rows)


def ablation_plan() -> dict[str, list[str]]:
    core = [
        "job_title",
        "job_category",
        "education_required",
        "city",
        "country",
        "remote_work",
        "company_size",
        "industry",
        "demand_score",
        "benefits_score_10",
    ]
    return {
        "A0_CONSERVATIVE_CORE": core,
        "A1_PLUS_YEARS": core + ["years_of_experience"],
        "A2_EXPERIENCE_BUCKET": core + ["experience_level"],
        "A6_SKILLS": core + ["years_of_experience", "required_skills", "skill_count"],
    }


def numeric_vif(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
    work = work.fillna(work.median(numeric_only=True))
    rows: list[dict[str, Any]] = []
    for feature in columns:
        others = [c for c in columns if c != feature]
        if not others or work[feature].nunique() <= 1:
            vif = 1.0
        else:
            model = LinearRegression().fit(work[others], work[feature])
            r2 = float(model.score(work[others], work[feature]))
            vif = float("inf") if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append({"feature": feature, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def dependency_report(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("job_title", "salary_min_usd"),
        ("job_title", "salary_max_usd"),
        ("job_title", "demand_score"),
        ("is_senior", "experience_level"),
        ("is_remote_friendly", "remote_work"),
        ("is_llm_role", "job_title"),
        ("city", "country"),
    ]
    rows: list[dict[str, Any]] = []
    for source, target in pairs:
        if source not in df.columns or target not in df.columns:
            continue
        grouped = df.groupby(source, dropna=False)[target].nunique(dropna=False)
        exact = bool((grouped <= 1).all())
        rows.append(
            {
                "source": source,
                "target": target,
                "source_unique": int(df[source].nunique(dropna=False)),
                "max_target_values_per_source": int(grouped.max()),
                "exact_functional_dependency": exact,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(data_path: Path | None = None, output_root: Path | None = None) -> PipelinePaths:
    project_root = Config.ROOT_DIR
    config = Config.load()
    paths = build_paths(project_root, data_path, output_root)
    np.random.seed(int(config["project"]["seed"]))

    # Clean outputs from previous run while keeping directory contract stable.
    for folder in [paths.basic, paths.ml_ready, paths.comparison, paths.best, paths.prediction]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STAGE 1 — LOAD DATA
    # ------------------------------------------------------------------
    stage_header(1, STAGES[0])
    if not paths.data_raw.exists():
        raise FileNotFoundError(f"Raw dataset not found: {paths.data_raw}")
    raw = pd.read_csv(paths.data_raw, low_memory=False)
    required = [TARGET, "job_id", "AI Engineering", "posting_year", "posting_month", "required_skills"]
    missing_required = [c for c in required if c not in raw.columns]
    if missing_required:
        raise RuntimeError(f"Schema gate failed; missing required columns: {missing_required}")
    manifest = {
        "file": str(paths.data_raw.relative_to(project_root)),
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "file_size_bytes": int(paths.data_raw.stat().st_size),
        "sha256": sha256_file(paths.data_raw),
        "period_min": f"{int(raw.posting_year.min())}-{int(raw.loc[raw.posting_year.eq(raw.posting_year.min()), 'posting_month'].min()):02d}",
        "period_max": f"{int(raw.posting_year.max())}-{int(raw.loc[raw.posting_year.eq(raw.posting_year.max()), 'posting_month'].max()):02d}",
    }
    save_json(manifest, paths.basic / "01_raw_manifest.json")
    save_csv(raw.head(25), paths.basic / "01_raw_preview.csv")

    # ------------------------------------------------------------------
    # STAGE 2 — SCOPE & INITIAL INSPECTION
    # ------------------------------------------------------------------
    stage_header(2, STAGES[1])
    dictionary = data_dictionary(raw)
    save_csv(dictionary, paths.basic / "02_data_dictionary_raw.csv")
    numeric_summary = raw.select_dtypes(include=np.number).describe().T.reset_index().rename(columns={"index": "column"})
    save_csv(numeric_summary, paths.basic / "02_numeric_summary_raw.csv")
    profile, outliers = target_profile(raw)
    save_csv(profile, paths.basic / "02_target_profile_raw.csv")
    save_csv(outliers, paths.basic / "02_target_outliers_iqr.csv")
    task = pd.DataFrame(
        [
            ["Task type", "Supervised regression"],
            ["Modelling unit", "One AI job posting / one valid CSV row"],
            ["Target y", TARGET],
            ["Input X", "Inference-time eligible features only"],
            ["Prediction moment", "Before annual salary is known/finalized"],
            ["Primary output", "Predicted annual salary in USD"],
            ["Secondary output", "Empirical 90% error interval + OOD/review flag + model drivers"],
        ],
        columns=["component", "definition"],
    )
    save_csv(task, paths.basic / "02_formal_task_definition.csv")
    plot_target_distribution(raw, paths.basic / "02_target_distribution_raw.png")
    plot_cardinality(dictionary, paths.basic / "02_feature_cardinality_raw.png")

    # ------------------------------------------------------------------
    # STAGE 3 — DATA QUALITY CHECK
    # ------------------------------------------------------------------
    stage_header(3, STAGES[2])
    hidden_tokens = {str(x).lower() for x in config["analysis"]["hidden_missing_tokens"]}
    quality_rows: list[dict[str, Any]] = []
    for column in raw.columns:
        series = raw[column]
        hidden = 0
        if series.dtype == "object":
            norm = series.dropna().astype(str).str.strip().str.lower()
            hidden = int(norm.isin(hidden_tokens).sum())
        quality_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "hidden_missing_tokens": hidden,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality_summary = pd.DataFrame(
        [
            {
                "rows": len(raw),
                "columns": raw.shape[1],
                "missing_cells": int(raw.isna().sum().sum()),
                "hidden_missing_tokens": int(quality["hidden_missing_tokens"].sum()),
                "duplicate_rows": int(raw.duplicated().sum()),
                "unique_job_id": int(raw["job_id"].nunique()),
                "invalid_job_category_token_rows": int(raw["AI Engineering"].astype(str).str.strip().eq("job_category").sum()),
            }
        ]
    )
    save_csv(quality, paths.basic / "03_data_quality_by_column.csv")
    save_csv(quality_summary, paths.basic / "03_data_quality_summary.csv")
    save_csv(categorical_value_audit(raw), paths.basic / "03_categorical_value_audit.csv")
    plot_missingness(quality, paths.basic / "03_missingness_by_feature.png")
    plot_cardinality(dictionary, paths.basic / "03_high_cardinality_features.png")

    # ------------------------------------------------------------------
    # STAGE 4 — CORRUPTED ROW REMOVAL
    # ------------------------------------------------------------------
    stage_header(4, STAGES[3])
    corruption_mask = raw["AI Engineering"].astype(str).str.strip().eq("job_category")
    corrupted = raw.loc[corruption_mask].copy()
    save_csv(corrupted, paths.basic / "04_corrupted_rows_removed.csv")
    clean = raw.loc[~corruption_mask].copy().rename(columns={"AI Engineering": "job_category"})
    for column in clean.select_dtypes(include="object").columns:
        clean[column] = clean[column].astype(str).str.strip()
    raw_skills_without_corruption = raw.loc[~corruption_mask, "required_skills"].fillna("")
    duplicate_skill_rows = raw_skills_without_corruption.map(
        lambda s: len([x.strip() for x in str(s).split("|") if x.strip()])
        != len(set(x.strip() for x in str(s).split("|") if x.strip()))
    )
    clean["required_skills"] = clean["required_skills"].map(normalize_skill_string)
    clean["skill_count"] = clean["required_skills"].map(skill_count)
    clean.reset_index(drop=True, inplace=True)
    save_csv(clean, paths.basic / "data_basic_clean.csv")
    stage4_summary = pd.DataFrame(
        [
            {
                "raw_rows": len(raw),
                "removed_corrupted_rows": int(corruption_mask.sum()),
                "clean_rows": len(clean),
                "duplicate_rows_after": int(clean.duplicated().sum()),
            }
        ]
    )
    save_csv(stage4_summary, paths.basic / "04_cleaning_summary.csv")

    # Lock March-2026 BEFORE target-aware Stage 5/7 diagnostics.
    locked_mask = clean["posting_year"].astype(int).eq(int(config["split"]["locked_test"]["year"])) & clean["posting_month"].astype(int).eq(int(config["split"]["locked_test"]["month"]))
    dev = clean.loc[~locked_mask].copy().sort_values(["posting_year", "posting_month"]).reset_index(drop=True)
    locked = clean.loc[locked_mask].copy().reset_index(drop=True)
    if dev.empty or locked.empty:
        raise RuntimeError("Temporal split gate failed: development or locked-test partition is empty.")

    # ------------------------------------------------------------------
    # STAGE 5 — CONTRADICTORY-FEATURE INVESTIGATION
    # ------------------------------------------------------------------
    stage_header(5, STAGES[4])
    expected_exp = clean["years_of_experience"].map(experience_bucket)
    exp_mismatch = clean["experience_level"].astype(str).ne(expected_exp.astype(str))
    range_bad_dev = ~dev[TARGET].between(dev["salary_min_usd"], dev["salary_max_usd"], inclusive="both")
    tier_bad_dev = dev["salary_tier"].astype(str).ne(dev[TARGET].map(salary_tier_expected).astype(str))
    issues = pd.DataFrame(
        [
            {
                "issue": "experience_bucket_mismatch",
                "affected_rows": int(exp_mismatch.sum()),
                "total_rows": len(clean),
                "affected_pct": float(exp_mismatch.mean() * 100),
                "scope": "all clean rows; target-free logic",
                "action": "Quarantine experience_level; retain years_of_experience only through explicit ablation and cautious interpretation.",
            },
            {
                "issue": "salary_outside_min_max",
                "affected_rows": int(range_bad_dev.sum()),
                "total_rows": len(dev),
                "affected_pct": float(range_bad_dev.mean() * 100),
                "scope": "development only",
                "action": "Block salary_min_usd and salary_max_usd from X.",
            },
            {
                "issue": "salary_tier_mismatch",
                "affected_rows": int(tier_bad_dev.sum()),
                "total_rows": len(dev),
                "affected_pct": float(tier_bad_dev.mean() * 100),
                "scope": "development only",
                "action": "Block salary_tier from X.",
            },
            {
                "issue": "skill_rows_with_duplicate_tokens",
                "affected_rows": int(duplicate_skill_rows.sum()),
                "total_rows": len(clean),
                "affected_pct": float(duplicate_skill_rows.mean() * 100),
                "scope": "all clean rows; target-free logic",
                "action": "De-duplicate tokens within each row before multi-hot encoding.",
            },
        ]
    )
    save_csv(issues, paths.basic / "05_logic_integrity_findings.csv")
    save_csv(clean.loc[exp_mismatch, ["job_id", "experience_level", "years_of_experience"]], paths.basic / "05_experience_mismatch_rows.csv")
    save_csv(dependency_report(clean), paths.basic / "05_functional_dependency_report.csv")
    by_level = (
        dev.groupby("experience_level")[TARGET]
        .agg(count="count", mean_salary_usd="mean", median_salary_usd="median")
        .reset_index()
    )
    by_year = (
        dev.groupby("years_of_experience")[TARGET]
        .agg(count="count", mean_salary_usd="mean", median_salary_usd="median")
        .reset_index()
    )
    by_domain = (
        dev.groupby("job_category")[TARGET]
        .agg(count="count", mean_salary_usd="mean", median_salary_usd="median")
        .reset_index()
        .sort_values("mean_salary_usd", ascending=False)
    )
    save_csv(by_level, paths.basic / "05_salary_by_experience_level_dev.csv")
    save_csv(by_year, paths.basic / "05_salary_by_years_dev.csv")
    save_csv(by_domain, paths.basic / "05_salary_by_job_category_dev.csv")
    job_suffix = pd.to_numeric(dev["job_id"].str.extract(r"(\d+)$")[0], errors="coerce")
    save_json(
        {
            "job_id_suffix_target_pearson_r_dev": float(job_suffix.corr(dev[TARGET])),
            "interpretation": "Identifier-ordering artifact; job_id is strictly blocked from X.",
        },
        paths.basic / "05_identifier_ordering_audit.json",
    )
    plot_logic_issue_rates(issues, paths.basic / "05_logic_issue_rates.png")
    plot_salary_by_job_category(by_domain, paths.basic / "05_salary_by_job_category.png")
    """ plot_years_by_experience_level(clean, paths.basic / "05_years_by_experience_level.png")
    plot_salary_by_experience_level(dev, paths.basic / "05_salary_by_experience_level.png")
    plot_salary_by_years(by_year, paths.basic / "05_salary_by_years_experience.png") """
    plot_experience_salary_dashboard(clean, by_level, by_year, paths.basic / "05_experience_salary_dashboard.png")
    #plot_salary_range_integrity(dev, paths.basic / "05_salary_range_vs_target.png")
    # Gọi hàm vẽ Dashboard Consistency (Ném data DEV vào để soi)
    plot_consistency_audit_dashboard(clean, paths.basic / "05_salary_consistency_dashboard.png")

    # ------------------------------------------------------------------
    # STAGE 6 — FEATURE SELECTION & LEAKAGE PREVENTION
    # ------------------------------------------------------------------
    stage_header(6, STAGES[5])
    policy = feature_policy_table()
    save_csv(policy, paths.ml_ready / "06_feature_policy.csv")
    save_csv(pd.DataFrame({"feature": MODEL_FEATURES}), paths.ml_ready / "06_primary_model_features.csv")
    plan = ablation_plan()
    save_csv(
        pd.DataFrame(
            [
                {
                    "experiment": name,
                    "feature_count": len(features),
                    "features": " | ".join(features),
                    "decision_question": {
                        "A0_CONSERVATIVE_CORE": "Reference plausibility without experience or skills.",
                        "A1_PLUS_YEARS": "Does granular years materially improve temporal validation?",
                        "A2_EXPERIENCE_BUCKET": "Does the contradictory categorical bucket add defensible signal?",
                        "A6_SKILLS": "Do TRAIN-only multi-hot skills justify added complexity?",
                    }[name],
                }
                for name, features in plan.items()
            ]
        ),
        paths.ml_ready / "06_ablation_plan.csv",
        
    )
    # Render Mockup Dashboard 6 ô cho Stage 06
    plot_planned_ablation_dashboard(paths.ml_ready / "06_ablation_dashboard.png")

    overlap = sorted(set(MODEL_FEATURES) & set(BLOCKED_FEATURES))
    if overlap:
        raise RuntimeError(f"Leakage gate failed: blocked fields entered X: {overlap}")
    save_json({"status": "PASS", "blocked_overlap": overlap}, paths.ml_ready / "06_leakage_gate.json")
    plot_feature_policy(policy, paths.ml_ready / "06_feature_governance_mix.png")

    # ------------------------------------------------------------------
    # STAGE 7 — CORRELATION ENCODING & ANALYSIS (DEV ONLY)
    # ------------------------------------------------------------------
    stage_header(7, STAGES[6])
    pre_analysis = build_preprocessor(MODEL_FEATURES, scale_numeric=True)
    x_dev_encoded = pre_analysis.fit_transform(dev[MODEL_FEATURES])
    feature_names = list(pre_analysis.get_feature_names_out())
    encoded = pd.DataFrame(x_dev_encoded, columns=feature_names)
    encoded.insert(0, "row_id", dev.index)
    save_csv(encoded, paths.ml_ready / "07_encoded_dev_analysis_only.csv")
    target_series = dev[TARGET].reset_index(drop=True)
    correlation_rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame(x_dev_encoded, columns=feature_names)
    for feature in feature_names:
        values = matrix[feature]
        if values.nunique(dropna=True) <= 1:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson = float(values.corr(target_series, method="pearson"))
            spearman = float(values.corr(target_series, method="spearman"))
        correlation_rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson),
                "abs_spearman": abs(spearman),
            }
        )
    corr = pd.DataFrame(correlation_rows).sort_values("abs_pearson", ascending=False).reset_index(drop=True)
    save_csv(corr, paths.ml_ready / "07_target_correlations_dev.csv")
    vif_columns = [c for c in ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"] if c in dev.columns]
    vif = numeric_vif(dev, vif_columns)
    save_csv(vif, paths.ml_ready / "07_vif_dev_numeric.csv")
    pairwise = matrix.corr(method="pearson")
    high_pairs: list[dict[str, Any]] = []
    threshold = float(config["analysis"]["pairwise_correlation_high"])
    for i, left in enumerate(pairwise.columns):
        for right in pairwise.columns[i + 1 :]:
            value = pairwise.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                high_pairs.append({"feature_1": left, "feature_2": right, "pearson": float(value), "abs_pearson": abs(float(value))})
    save_csv(pd.DataFrame(high_pairs).sort_values("abs_pearson", ascending=False) if high_pairs else pd.DataFrame(columns=["feature_1", "feature_2", "pearson", "abs_pearson"]), paths.ml_ready / "07_high_pairwise_correlations_dev.csv")

    # Skill evidence learned only from development data.
    skill_rows: list[dict[str, Any]] = []
    for skill in pre_analysis.named_transformers_["skills"].get_feature_names_out():
        mask = dev["required_skills"].fillna("").map(lambda s, token=skill: token in set(str(s).split("|")))
        subset = dev.loc[mask]
        skill_rows.append(
            {
                "skill_token": str(skill),
                "record_count": int(mask.sum()),
                "mean_salary_usd": float(subset[TARGET].mean()) if len(subset) else np.nan,
                "median_salary_usd": float(subset[TARGET].median()) if len(subset) else np.nan,
                "mean_years_of_experience": float(subset["years_of_experience"].mean()) if len(subset) else np.nan,
            }
        )
    skill_summary = pd.DataFrame(skill_rows).sort_values("record_count", ascending=False).reset_index(drop=True)
    save_csv(skill_summary, paths.ml_ready / "07_skill_frequency_and_salary_dev.csv")
    save_csv(skill_summary.loc[skill_summary["record_count"] < 50], paths.ml_ready / "07_skills_under_50_records_dev.csv")

    raw_target = dev[TARGET].astype(float)
    log_target = np.log1p(raw_target)
    transform_review = pd.DataFrame(
        [
            {"target_version": "raw_usd", "skewness": float(raw_target.skew()), "mean": float(raw_target.mean()), "std": float(raw_target.std()), "modeling_decision": "Primary evaluation remains on original USD scale."},
            {"target_version": "log1p_diagnostic", "skewness": float(pd.Series(log_target).skew()), "mean": float(np.mean(log_target)), "std": float(np.std(log_target, ddof=1)), "modeling_decision": "Diagnostic only; not forced across all model families without temporal-CV evidence."},
        ]
    )
    save_csv(transform_review, paths.ml_ready / "07_target_transform_review.csv")
    plot_correlations(corr, paths.ml_ready / "07_top30_correlations.png", top_n=30)
    plot_pearson_spearman(corr, paths.ml_ready / "07_pearson_vs_spearman.png", top_n=15)
    plot_vif(vif, paths.ml_ready / "07_vif_numeric.png")
    plot_skills(skill_summary, paths.ml_ready / "07_top30_skills.png", top_n=30)

    # ------------------------------------------------------------------
    # STAGE 8 — TEMPORAL SPLIT & TRAINING READINESS
    # ------------------------------------------------------------------
    stage_header(8, STAGES[7])
    split_summary = pd.DataFrame(
        [
            {"partition": "TRAIN_DEV", "rows": len(dev), "pct": len(dev) / len(clean) * 100, "rule": "Before 2026-03"},
            {"partition": "LOCKED_TEST", "rows": len(locked), "pct": len(locked) / len(clean) * 100, "rule": "2026-03 only"},
        ]
    )
    save_csv(split_summary, paths.ml_ready / "08_split_summary.csv")
    monthly = clean.groupby(["posting_year", "posting_month"]).size().reset_index(name="rows")
    monthly["period"] = monthly["posting_year"].astype(int).astype(str) + "-" + monthly["posting_month"].astype(int).astype(str).str.zfill(2)
    monthly["partition"] = np.where((monthly["posting_year"].eq(LOCKED_YEAR)) & (monthly["posting_month"].eq(LOCKED_MONTH)), "LOCKED_TEST", "TRAIN_DEV")
    save_csv(monthly, paths.ml_ready / "08_monthly_distribution.csv")
    before_after = pd.DataFrame(
        [
            ["Rows", f"{len(raw):,} raw", f"{len(clean):,} clean"],
            ["Features", "25 raw columns", f"{len(MODEL_FEATURES)} serving fields including skills"],
            ["Corruption", "Header token present", "Confirmed row removed"],
            ["Experience", "Contradictory pair", "experience_level quarantined; years tested by ablation"],
            ["Skills", "Unique pipe-delimited strings", f"{len(skill_summary):,} DEV-only tokens + skill_count"],
            ["Leakage", "salary_min/max/tier present", "Blocked from X"],
            ["Encoding", "Raw categories/text", "ColumnTransformer + one-hot/ordinal/RobustScaler/multi-hot"],
            ["Evaluation", "No governed holdout", "March-2026 locked final test"],
        ],
        columns=["aspect", "before", "after"],
    )
    save_csv(before_after, paths.ml_ready / "08_before_after_processing.csv")
    deploy_pre = build_preprocessor(MODEL_FEATURES, scale_numeric=False)
    x_train_encoded = deploy_pre.fit_transform(dev[MODEL_FEATURES])
    x_test_encoded = deploy_pre.transform(locked[MODEL_FEATURES])
    deploy_names = list(deploy_pre.get_feature_names_out())
    save_csv(pd.DataFrame(x_train_encoded, columns=deploy_names), paths.ml_ready / "08_X_train_encoded.csv")
    save_csv(pd.DataFrame(x_test_encoded, columns=deploy_names), paths.ml_ready / "08_X_locked_test_encoded.csv")
    save_csv(pd.DataFrame({TARGET: dev[TARGET].to_numpy()}), paths.ml_ready / "08_y_train.csv")
    save_csv(pd.DataFrame({TARGET: locked[TARGET].to_numpy()}), paths.ml_ready / "08_y_locked_test.csv")
    joblib.dump(deploy_pre, paths.artifacts / "preprocessor_ml_ready.joblib")
    save_json(
        {
            "train_rows": len(dev),
            "locked_test_rows": len(locked),
            "encoded_feature_count": len(deploy_names),
            "skill_vocabulary_count": int(len(deploy_pre.named_transformers_["skills"].get_feature_names_out())),
            "fit_scope": "TRAIN/DEV only; locked test transformed only",
        },
        paths.ml_ready / "08_training_readiness.json",
    )
    plot_temporal_split(monthly, paths.ml_ready / "08_temporal_split_timeline.png")

    # ------------------------------------------------------------------
    # STAGE 9 — MODEL TRAINING & TEMPORAL CV COMPARISON
    # ------------------------------------------------------------------
    stage_header(9, STAGES[8])
    folds = monthly_temporal_folds(dev, n_folds=int(config["split"]["temporal_cv_folds"]))
    fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for definition in model_catalog():
        model_fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            tr = dev.loc[fold["train_idx"]]
            va = dev.loc[fold["validation_idx"]]
            pipe = Pipeline(
                [
                    ("preprocessor", build_preprocessor(MODEL_FEATURES, definition.scale_numeric)),
                    ("model", definition.factory()),
                ]
            )
            start = time.perf_counter()
            pipe.fit(tr[MODEL_FEATURES], tr[TARGET])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            pred = pipe.predict(va[MODEL_FEATURES])
            predict_seconds = time.perf_counter() - start
            metrics = regression_metrics(va[TARGET], pred)
            row = {
                **metrics,
                "model": definition.name,
                "is_baseline": definition.is_baseline,
                "fold": fold["fold"],
                "validation_month": str(fold["validation_month"]),
                "train_rows": len(tr),
                "validation_rows": len(va),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
            fold_rows.append(row)
            model_fold_metrics.append(row)
        fm = pd.DataFrame(model_fold_metrics)
        comparison_rows.append(
            {
                "model": definition.name,
                "is_baseline": definition.is_baseline,
                "CV_MAE_mean": float(fm["MAE"].mean()),
                "CV_MAE_std": float(fm["MAE"].std(ddof=0)),
                "CV_RMSE_mean": float(fm["RMSE"].mean()),
                "CV_RMSE_std": float(fm["RMSE"].std(ddof=0)),
                "CV_R2_mean": float(fm["R2"].mean()),
                "CV_R2_std": float(fm["R2"].std(ddof=0)),
                "CV_MedAE_mean": float(fm["MedAE"].mean()),
                "fit_seconds_mean": float(fm["fit_seconds"].mean()),
                "predict_seconds_mean": float(fm["predict_seconds"].mean()),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    comparison = pd.DataFrame(comparison_rows).sort_values(["CV_MAE_mean", "CV_RMSE_mean"]).reset_index(drop=True)
    save_csv(fold_df, paths.comparison / "09_model_comparison_fold_metrics.csv")
    save_csv(comparison, paths.comparison / "09_model_comparison_temporal_cv.csv")
    plot_model_comparison(comparison, paths.comparison)
    plot_fold_stability(fold_df, paths.comparison / "09_fold_stability_mae.png")

    # Execute required feature-family ablation with a stable RF comparator.
    ablation_rows: list[dict[str, Any]] = []
    for name, feature_set in plan.items():
        fold_metrics: list[dict[str, float]] = []
        for fold in folds:
            tr = dev.loc[fold["train_idx"]]
            va = dev.loc[fold["validation_idx"]]
            estimator = RandomForestRegressor(
                n_estimators=60,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=SEED,
                n_jobs=1,
            )
            pipe = Pipeline([("preprocessor", build_preprocessor(feature_set, False)), ("model", estimator)])
            pipe.fit(tr[feature_set], tr[TARGET])
            pred = pipe.predict(va[feature_set])
            fold_metrics.append(regression_metrics(va[TARGET], pred))
        af = pd.DataFrame(fold_metrics)
        ablation_rows.append(
            {
                "experiment": name,
                "feature_count_raw": len(feature_set),
                "CV_MAE_mean": float(af["MAE"].mean()),
                "CV_RMSE_mean": float(af["RMSE"].mean()),
                "CV_R2_mean": float(af["R2"].mean()),
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    base_mae = float(ablation.loc[ablation["experiment"].eq("A0_CONSERVATIVE_CORE"), "CV_MAE_mean"].iloc[0])
    ablation["MAE_improvement_vs_A0_pct"] = (base_mae - ablation["CV_MAE_mean"]) / base_mae * 100
    save_csv(ablation.sort_values("CV_MAE_mean"), paths.comparison / "09_feature_family_ablation.csv")
    plot_ablation(ablation, paths.comparison / "09_feature_family_ablation.png")

    candidates = comparison.loc[~comparison["is_baseline"].astype(bool)].copy()
    selected_name = str(candidates.iloc[0]["model"])

    # Feature-importance stability across folds for selected family (permutation on validation months).
    selected_definition = next(item for item in model_catalog() if item.name == selected_name)
    drift_rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = dev.loc[fold["train_idx"]]
        va = dev.loc[fold["validation_idx"]]
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(MODEL_FEATURES, selected_definition.scale_numeric)),
                ("model", selected_definition.factory()),
            ]
        )
        pipe.fit(tr[MODEL_FEATURES], tr[TARGET])
        pi = permutation_importance(
            pipe,
            va[MODEL_FEATURES],
            va[TARGET],
            scoring="neg_mean_absolute_error",
            n_repeats=1,
            random_state=SEED,
            n_jobs=1,
        )
        for feature, importance in zip(MODEL_FEATURES, pi.importances_mean):
            drift_rows.append({"fold": fold["fold"], "validation_month": fold["validation_month"], "raw_feature": feature, "importance": float(importance)})
    drift = pd.DataFrame(drift_rows)
    save_csv(drift, paths.comparison / "09_feature_importance_by_fold.csv")
    plot_importance_drift(drift, paths.comparison / "09_feature_importance_drift.png", top_n=10)

    # ------------------------------------------------------------------
    # STAGE 10 — BEST MODEL, TUNING, FINAL LOCKED TEST, IMPORTANCE
    # ------------------------------------------------------------------
    stage_header(10, STAGES[9])
    if selected_name == "Random Forest":
        tuned_params, tuning = tune_random_forest(dev, folds, MODEL_FEATURES)
        best_estimator = RandomForestRegressor(random_state=SEED, n_jobs=1, **tuned_params)
        scale_best = False
    elif selected_name == "Gradient Boosting":
        tuned_params, tuning = tune_gradient_boosting(dev, folds, MODEL_FEATURES)
        best_estimator = GradientBoostingRegressor(random_state=SEED, **tuned_params)
        scale_best = False
    else:
        tuned_params = selected_definition.factory().get_params()
        tuning = pd.DataFrame([{"candidate_id": 1, "CV_MAE_mean": float(candidates.iloc[0]["CV_MAE_mean"]), "note": "No extra tuning grid for this family."}])
        best_estimator = clone(selected_definition.factory())
        scale_best = selected_definition.scale_numeric
    save_csv(tuning, paths.best / "10_best_model_tuning_results.csv")

    best_pipe = Pipeline([("preprocessor", build_preprocessor(MODEL_FEATURES, scale_best)), ("model", best_estimator)])
    best_pipe.fit(dev[MODEL_FEATURES], dev[TARGET])
    locked_pred = best_pipe.predict(locked[MODEL_FEATURES])
    final_metrics = regression_metrics(locked[TARGET], locked_pred)
    final_metrics_row = {
        "selected_model": selected_name,
        **final_metrics,
        "selection_basis": "Lowest mean temporal-CV MAE among non-dummy candidates; locked test opened once after selection/tuning.",
    }
    save_csv(pd.DataFrame([final_metrics_row]), paths.best / "10_final_locked_test_metrics.csv")

    fitted_pre = best_pipe.named_steps["preprocessor"]
    encoded_names = list(fitted_pre.get_feature_names_out())
    fitted_model = best_pipe.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importance_values = np.asarray(fitted_model.feature_importances_, dtype=float)
    elif hasattr(fitted_model, "coef_"):
        importance_values = np.abs(np.ravel(fitted_model.coef_))
    else:
        importance_values = np.full(len(encoded_names), np.nan)
    encoded_imp = pd.DataFrame({"encoded_feature": encoded_names, "importance": importance_values}).sort_values("importance", ascending=False).reset_index(drop=True)
    save_csv(encoded_imp, paths.best / "10_encoded_feature_importance.csv")

    perm = permutation_importance(
        best_pipe,
        locked[MODEL_FEATURES],
        locked[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1,
    )
    raw_imp = pd.DataFrame(
        {
            "raw_feature": MODEL_FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    save_csv(raw_imp, paths.best / "10_raw_feature_permutation_importance.csv")
    plot_best_outputs(locked[TARGET], locked_pred, encoded_imp, raw_imp, paths.best)

    locked_eval = locked[["job_title", "job_category", "city", "country", "experience_level", TARGET]].copy()
    locked_eval["predicted_salary_usd"] = locked_pred
    locked_eval["absolute_error_usd"] = np.abs(locked_eval[TARGET] - locked_eval["predicted_salary_usd"])
    save_csv(locked_eval, paths.best / "10_locked_test_predictions_with_error.csv")
    segment_rows: list[dict[str, Any]] = []
    min_segment = int(config["analysis"]["min_segment_rows"])
    for segment_column in ["job_category", "country", "experience_level"]:
        for value, group in locked_eval.groupby(segment_column):
            if len(group) < min_segment:
                continue
            metrics = regression_metrics(group[TARGET], group["predicted_salary_usd"].to_numpy())
            segment_rows.append({"segment_feature": segment_column, "segment_value": value, "rows": len(group), **metrics})
    save_csv(pd.DataFrame(segment_rows), paths.best / "10_error_slices.csv")

    # Out-of-fold absolute residual distribution for a practical empirical prediction interval.
    oof_abs_errors: list[float] = []
    for fold in folds:
        tr = dev.loc[fold["train_idx"]]
        va = dev.loc[fold["validation_idx"]]
        fold_pipe = Pipeline([("preprocessor", build_preprocessor(MODEL_FEATURES, scale_best)), ("model", clone(best_estimator))])
        fold_pipe.fit(tr[MODEL_FEATURES], tr[TARGET])
        pred = fold_pipe.predict(va[MODEL_FEATURES])
        oof_abs_errors.extend(np.abs(va[TARGET].to_numpy() - pred).tolist())
    interval_q90 = float(np.quantile(oof_abs_errors, float(config["prediction"]["interval_quantile"])))

    # ------------------------------------------------------------------
    # STAGE 11 — SAVE DEPLOYABLE PIPELINE + METADATA
    # ------------------------------------------------------------------
    stage_header(11, STAGES[10])
    joblib.dump(best_pipe, paths.artifacts / "model_bundle.joblib")
    joblib.dump(fitted_pre, paths.artifacts / "preprocessor.pkl")
    joblib.dump(fitted_model, paths.artifacts / "model.pkl")
    save_json({"feature_columns": MODEL_FEATURES}, paths.artifacts / "feature_columns.json")
    category_features = ["job_title", "job_category", "education_required", "city", "country", "remote_work", "company_size", "industry"]
    category_options = {c: sorted(dev[c].dropna().astype(str).unique().tolist()) for c in category_features}
    numeric_features = ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
    numeric_ranges = {
        c: {"min": float(dev[c].min()), "max": float(dev[c].max()), "median": float(dev[c].median())}
        for c in numeric_features
    }
    skill_vocab_final = list(fitted_pre.named_transformers_["skills"].get_feature_names_out())
    metadata = {
        "project_name": "AI Job Market Salary Prediction",
        "target": TARGET,
        "model_name": selected_name,
        "model_features": MODEL_FEATURES,
        "structured_features": STRUCTURED_FEATURES,
        "blocked_features": BLOCKED_FEATURES,
        "locked_test": {"year": LOCKED_YEAR, "month": LOCKED_MONTH, "rows": len(locked)},
        "development_rows": len(dev),
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "education_order": EDUCATION_ORDER,
        "company_size_order": COMPANY_SIZE_ORDER,
        "skill_vocabulary": skill_vocab_final,
        "skill_vocabulary_count": len(skill_vocab_final),
        "prediction_interval_abs_error_q90": interval_q90,
        "final_locked_test_metrics": final_metrics,
        "selected_hyperparameters": tuned_params,
        "limitations": [
            "The supplied dataset contains contradictory and synthetic-looking relationships.",
            "Feature importance explains fitted model behavior, not causal salary economics.",
            "External validation on verified live job postings is required before operational compensation decisions.",
            "Cost-of-living geographic adjustment is not implemented because no verified external cost-of-living index is supplied.",
        ],
        "demo_login": {
            "username": str(config["prediction"]["demo_username"]),
            "password": str(config["prediction"]["demo_password"]),
            "warning": "Local academic demo only; replace with environment variables or Streamlit secrets before shared deployment.",
        },
    }
    save_json(metadata, paths.artifacts / "metadata.json")

    reloaded = joblib.load(paths.artifacts / "model_bundle.joblib")
    sample = dev[MODEL_FEATURES].head(12)
    before = best_pipe.predict(sample)
    after = reloaded.predict(sample)
    max_abs_difference = float(np.max(np.abs(before - after))) if len(before) else 0.0
    equivalence = {
        "status": "PASS" if np.allclose(before, after, rtol=1e-12, atol=1e-9) else "FAIL",
        "sample_rows": len(sample),
        "max_abs_prediction_difference": max_abs_difference,
    }
    save_json(equivalence, paths.artifacts / "11_bundle_equivalence.json")
    if equivalence["status"] != "PASS":
        raise RuntimeError("Deployment equivalence gate failed.")
    save_csv(
        pd.DataFrame(
            [
                ["model_bundle.joblib", "Full preprocessing + estimator pipeline used by Streamlit"],
                ["preprocessor.pkl", "Fitted preprocessing object"],
                ["model.pkl", "Final estimator fitted on preprocessed DEV features"],
                ["feature_columns.json", "Exact raw serving feature contract"],
                ["metadata.json", "Category options, numeric ranges, skills, interval and limitations"],
                ["11_bundle_equivalence.json", "Reloaded-bundle numerical-equivalence gate"],
            ],
            columns=["artifact", "purpose"],
        ),
        paths.best / "11_deployment_artifact_manifest.csv",
    )

    # ------------------------------------------------------------------
    # STAGE 12 — STREAMLIT REPORT CONTRACT + PREDICTION OUTPUT
    # ------------------------------------------------------------------
    stage_header(12, STAGES[11])
    pred_df = locked[["job_title", "job_category", "city", "country", TARGET]].copy()
    pred_df["predicted_salary_usd"] = locked_pred
    pred_df["absolute_error_usd"] = np.abs(pred_df[TARGET] - pred_df["predicted_salary_usd"])
    pred_df["prediction_low_90"] = np.maximum(0, pred_df["predicted_salary_usd"] - interval_q90)
    pred_df["prediction_high_90"] = pred_df["predicted_salary_usd"] + interval_q90
    save_csv(pred_df, paths.prediction / "12_locked_test_prediction_examples.csv")
    prediction_summary = pd.DataFrame(
        [
            {
                "model": selected_name,
                "locked_test_rows": len(pred_df),
                "prediction_mean_usd": float(pred_df["predicted_salary_usd"].mean()),
                "prediction_median_usd": float(pred_df["predicted_salary_usd"].median()),
                "interval_half_width_q90_usd": interval_q90,
                **final_metrics,
            }
        ]
    )
    save_csv(prediction_summary, paths.prediction / "12_prediction_summary.csv")
    for src_name, dst_name in [
        ("actual_vs_predicted_locked_test.png", "salary_prediction_actual_vs_predicted.png"),
        ("locked_test_residuals.png", "salary_prediction_residuals.png"),
    ]:
        source = paths.best / src_name
        if source.exists():
            shutil.copy2(source, paths.prediction / dst_name)

    plot_pipeline_12(paths.assets / "12_stage_pipeline.png")
    report_index = {
        "01_data_basic_clean": str(paths.basic.relative_to(project_root)),
        "02_data_ready_for_machine_learning": str(paths.ml_ready.relative_to(project_root)),
        "03_model_comparison": str(paths.comparison.relative_to(project_root)),
        "04_best_model_and_feature_importance": str(paths.best.relative_to(project_root)),
        "05_salary_prediction": str(paths.prediction.relative_to(project_root)),
        "model_bundle": str((paths.artifacts / "model_bundle.joblib").relative_to(project_root)),
        "metadata": str((paths.artifacts / "metadata.json").relative_to(project_root)),
        "pipeline_picture": str((paths.assets / "12_stage_pipeline.png").relative_to(project_root)),
    }
    save_json(report_index, paths.output / "report_index.json")
    save_json(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seed": SEED,
            "raw_sha256": manifest["sha256"],
            "rows_raw": len(raw),
            "rows_clean": len(clean),
            "development_rows": len(dev),
            "locked_test_rows": len(locked),
            "selected_model": selected_name,
            "run_status": "SUCCESS",
        },
        paths.output / "pipeline_run_manifest.json",
    )
    print("\nPipeline completed successfully.")
    print(f"Best model: {selected_name}")
    print(f"Locked-test metrics: {json.dumps(final_metrics, indent=2)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Job Market Salary Prediction — 12-stage pipeline")
    parser.add_argument("--data", type=Path, default=None, help="Optional raw CSV path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output root")
    parser.add_argument("--log-level", default="INFO", help="Accepted for CLI compatibility")
    args = parser.parse_args()
    run_pipeline(args.data, args.output)
