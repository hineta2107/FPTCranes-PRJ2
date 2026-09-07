from __future__ import annotations

import pandas as pd


def stage5_experience_insight(issues: pd.DataFrame, by_level: pd.DataFrame, by_year: pd.DataFrame) -> tuple[str, str, str]:
    if issues.empty:
        return "No contradiction evidence available.", "Stage 5 outputs are incomplete.", "Re-run the pipeline."
    row = issues.loc[issues["issue"].eq("experience_bucket_mismatch")]
    mismatch = float(row.iloc[0]["affected_pct"]) if not row.empty else float("nan")
    level_spread = float(by_level["mean_salary_usd"].max() - by_level["mean_salary_usd"].min()) if not by_level.empty else 0.0
    year_spread = float(by_year["mean_salary_usd"].max() - by_year["mean_salary_usd"].min()) if not by_year.empty else 0.0
    finding = f"Experience semantics disagree in {mismatch:.1f}% of clean rows."
    if year_spread > max(level_spread * 3, 1):
        interpretation = (
            "years_of_experience carries a much stronger salary pattern than experience_level, while the categorical buckets remain comparatively flat. "
            "The pair should be treated as a contradictory feature family rather than two independent truths."
        )
        action = "Use ablation. Keep years_of_experience in the primary candidate only with a domain-validation warning; quarantine experience_level."
    else:
        interpretation = "The two experience views have similar aggregate salary spread, but their row-level semantic mismatch still requires ablation."
        action = "Do not include both by default; compare them under temporal CV."
    return finding, interpretation, action


def model_generalization_insight(comparison: pd.DataFrame, final_metrics: pd.DataFrame) -> str:
    if comparison.empty or final_metrics.empty:
        return "Model generalization evidence is unavailable."
    selected = str(final_metrics.iloc[0]["selected_model"])
    cv = comparison.loc[comparison["model"].eq(selected), "CV_MAE_mean"]
    if cv.empty:
        return f"{selected} was selected, but its CV MAE is unavailable."
    cv_mae = float(cv.iloc[0])
    test_mae = float(final_metrics.iloc[0]["MAE"])
    gap = (test_mae - cv_mae) / cv_mae * 100 if cv_mae else 0.0
    status = "material degradation" if gap > 20 else "moderate degradation" if gap > 10 else "good consistency"
    return f"Locked-test MAE is {gap:+.1f}% versus mean temporal-CV MAE for {selected}, indicating {status} on the March-2026 future holdout."
