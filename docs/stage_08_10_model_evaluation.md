# C. DESIGN PROCESS (continued)

> **Provenance**: All values sourced from pipeline run on 2026-08-23
> (`outputs/run_summary.json`, status: SUCCESS, elapsed: 34.58s).
> Artifact paths: `outputs/03_model_comparison/`, `outputs/04_best_model_and_feature_importance/`,
> `artifacts/metadata.json`.

---

## STAGE 8. Train–Test Split

(PIC = Minh)

### 8.1. Temporal Split Definition

The split is **temporal, not random**. The locked test period is all records
with `posting_year = 2026` and `posting_month = 3` (March 2026). Everything
strictly before that date forms the development (training) partition.

| Partition | Filter | Rows | Columns |
|-----------|--------|------|---------|
| TRAIN / DEV | posting date < 2026-03 | 1,201 | 13 features (11 structured + required_skills multi-hot + skill_count) |
| Locked test | posting_year = 2026, posting_month = 3 | 298 | Same schema as TRAIN |

The split ratio is approximately **80 / 20**, but the *identity* of the split
(March 2026) is what matters, not the ratio.

### 8.2. Locked-Test Rule

- The 2026-03 partition is **never** used inside cross-validation.
- Its labels are **not inspected** until *all* of the following are frozen:
  feature set, model class, hyperparameters, interval policy, and the
  narrative template used to report results.
- Any inadvertent use of test data during selection/tuning invalidates the
  final performance estimate; the run must be discarded and rebuilt.

### 8.3. Pipeline Requirement

- The split is performed on the **raw cleaned DataFrame** (`df`), *not* on
  the `pd.get_dummies` correlation view (`df_corr`).
- Each candidate model is wrapped in a single scikit-learn `Pipeline`
  containing a `ColumnTransformer` preprocessor:
  - `OrdinalEncoder` for `education_required`
  - `OneHotEncoder(handle_unknown="ignore")` for the other nominal features
  - Passthrough for numeric features (`years_of_experience`, `demand_score`,
    `benefits_score_10`, `skill_count`)
- The saved artifact is therefore the **entire pipeline — encoder and model
  together**.

### 8.4. Before vs After Data Processing

| Aspect | Before | After (required state before Stage 9) |
|--------|--------|---------------------------------------|
| Data structure | 25 raw columns incl. IDs, leakage, free text | 13 modelling features, leakage-free |
| Corrupted values | Header value leaked into a data row | Row removed, field renamed to `job_category` |
| Experience signals | Two contradictory columns (`years_of_experience` + `experience_level`) | Redundant column dropped |
| Encoding | Unsafe `get_dummies` (analysis only) | Deployable `ColumnTransformer` inside Pipeline |
| Target leakage | Present (`salary_min_usd`, `salary_max_usd`, `salary_tier`) | Fully eliminated |
| Model readiness | ❌ Not ready | ✅ Ready — single deployable artifact |

### 8.5. Failure-Mode Gate

| Failure mode | Required remediation |
|--------------|----------------------|
| Preprocessing fitted before split | Invalidate run and rebuild from raw data |
| Locked test used during tuning | Invalidate final performance estimate |
| Unknown category at inference | Encode safely via `handle_unknown="ignore"`, flag, monitor rate |
| Out-of-range numeric input | Predict only if policy allows; attach OOD flag |
| Bundle reload changes prediction | Release fails; do not serve |

---

## STAGE 9. Model Training & Comparison

(basic + ensemble model)

### 9.1. Candidate Model Ladder

Six regression models were trained and compared on the identical temporal
train/test split:

| # | Model | Role | Rationale |
|---|-------|------|-----------|
| 0 | Dummy (Median) | Non-ML floor | Any promoted model must beat it materially |
| 1 | Linear Regression | Transparent linear baseline | Detect additive relationships |
| 2 | Ridge Regression | Regularized linear baseline | Stable under many one-hot columns |
| 3 | Random Forest | Nonlinear bagging baseline | Robust tabular comparator |
| 4 | Gradient Boosting | Primary advanced candidate | Strong nonlinear tabular baseline |
| 5 | SVR (RBF kernel) | Diagnostic | Expected to underperform on sparse OHE |

### 9.2. Hyperparameter Tuning Rules

1. Tuning is nested inside the **development data only**, using expanding
   monthly temporal folds (5 folds: 2025-10 through 2026-02).
2. Use `RandomizedSearchCV` with a small bounded search space — the dev set
   has only ~1,201 rows.
3. Optimize primary metric = **negative MAE**, while also recording RMSE, R²,
   and fold variance.
4. Every trial logs: parameters, fold IDs, seed, and metrics.
5. Prefer the **simpler, more stable, more interpretable** candidate when two
   models are statistically/practically similar.

### 9.3. Tuning Results (Random Forest — Winner)

| Candidate | n_estimators | min_samples_leaf | max_features | max_depth | CV MAE mean (USD) | CV MAE std |
|-----------|-------------|-----------------|-------------|-----------|-------------------|-----------|
| 1 | 300 | 1 | 0.8 | — | 12,641 | 3,260 |
| **2 (selected)** | **400** | **2** | **0.8** | **—** | **12,540** | **3,208** |
| 3 | 400 | 2 | 0.6 | — | 12,686 | 3,076 |
| 4 | 400 | 1 | 0.8 | 16 | 12,722 | 3,185 |

Selected hyperparameters: `n_estimators=400`, `min_samples_leaf=2`,
`max_features=0.8`, `max_depth=None`.

### 9.4. Model Comparison — Temporal Cross-Validation Results

![Model Comparison — CV MAE](../outputs/03_model_comparison/model_comparison_cv_mae.png)

![Model Comparison — CV R²](../outputs/03_model_comparison/model_comparison_cv_r2.png)

| Rank | Model | CV MAE mean (USD) | CV MAE std | CV RMSE mean (USD) | CV RMSE std | CV R² mean | CV R² std | CV MedAE mean (USD) |
|------|-------|-------------------|-----------|--------------------|-----------|-----------|-----------|--------------------|
| 1 | Random Forest | 12,554 | 3,180 | 23,018 | 6,755 | 0.877 | 0.050 | 4,908 |
| 2 | Gradient Boosting | 13,372 | 1,399 | 25,333 | 3,816 | 0.856 | 0.023 | 6,200 |
| 3 | Ridge Regression | 25,699 | 3,154 | 35,442 | 3,897 | 0.715 | 0.045 | 17,558 |
| 4 | Linear Regression | 28,964 | 3,419 | 38,862 | 4,501 | 0.654 | 0.077 | 21,025 |
| 5 | SVR (RBF) | 53,409 | 5,112 | 67,735 | 5,911 | −0.034 | 0.038 | 42,721 |
| 6 | Dummy (Median) | 54,177 | 5,071 | 68,493 | 5,892 | −0.058 | 0.037 | 43,600 |

### 9.5. Evaluation Metrics

| Metric | Formula | Priority | Interpretation |
|--------|---------|----------|----------------|
| MAE | (1/n) · Σ \|yᵢ − ŷᵢ\| | **Primary** | Average absolute USD error |
| RMSE | √((1/n) · Σ (yᵢ − ŷᵢ)²) | Primary-secondary | Penalizes large misses |
| R² | 1 − SS_res / SS_tot | Diagnostic | Variance explained |
| MedAE | median(\|yᵢ − ŷᵢ\|) | Robust diagnostic | Typical error, outlier-robust |

### 9.6. Key Observations

**SVR structural mismatch (not a tuning failure):**
SVR with an RBF kernel measures distance in feature space. After one-hot
encoding, this dataset has approximately 189 dimensions, most of them sparse
binary indicators. Distance-based methods degrade badly in exactly this
setting — all points become roughly equidistant — while tree-based splits are
unaffected by how many sparse dummy columns exist. This explains why SVR
lands at or below the Dummy baseline, not a tuning mistake.

**`years_of_experience` caveat:**
The `years_of_experience` feature carries a contradictory, likely-synthetic
signal identified in Stages 5–6 (negative correlation with salary despite
logical expectation of positive correlation). Its strong model reliance
(see Stage 10) must be interpreted as *the model learned the pattern in this
dataset*, not as causal salary evidence. The R² score alone must not be
reported without inspecting feature importance.

---

## STAGE 10. Best Model Selection & Feature Importance Review

### 10.1. Model Selection

**Selected model**: Random Forest

**Selection basis**: Lowest mean temporal CV MAE on the development period;
locked test opened once after all selection and tuning decisions were frozen.

When Random Forest and Gradient Boosting are compared:

- Random Forest: CV MAE = 12,554 USD (std 3,180)
- Gradient Boosting: CV MAE = 13,372 USD (std 1,399)

Random Forest wins on primary metric. Although Gradient Boosting has lower
variance, the absolute MAE difference (818 USD) favors Random Forest, and
the simpler-model preference applies (Random Forest is more interpretable
via direct feature importance).

### 10.2. Locked-Test Performance

![Actual vs Predicted — Locked Test](../outputs/04_best_model_and_feature_importance/actual_vs_predicted_locked_test.png)

![Locked-Test Residuals](../outputs/04_best_model_and_feature_importance/locked_test_residuals.png)

| Metric | Value |
|--------|-------|
| MAE | 14,961 USD |
| RMSE | 29,844 USD |
| R² | 0.803 |
| MedAE | 4,467 USD |
| Selection basis | Lowest mean temporal CV MAE on development period; locked test opened once after selection/tuning. |

### 10.3. Encoded Feature Importance (Top 10)

![Top 25 Encoded Feature Importance](../outputs/04_best_model_and_feature_importance/top25_encoded_feature_importance.png)

From the fitted Random Forest estimator (impurity-based importance):

| Rank | Encoded Feature | Importance |
|------|----------------|-----------|
| 1 | `nominal__job_category_AI Engineering` | 0.5908 |
| 2 | `numeric__years_of_experience` | 0.2753 |
| 3 | `nominal__job_category_Robotics` | 0.0136 |
| 4 | `nominal__job_category_Security` | 0.0091 |
| 5 | `numeric__demand_score` | 0.0054 |
| 6 | `nominal__job_category_Architecture` | 0.0050 |
| 7 | `numeric__benefits_score_10` | 0.0044 |
| 8 | `education__education_required` | 0.0042 |
| 9 | `numeric__skill_count` | 0.0041 |
| 10 | `nominal__job_category_Business` | 0.0041 |

**Concentration**: `job_category_AI Engineering` (59.1%) +
`years_of_experience` (27.5%) = **86.6%** of total encoded feature
importance. The model is essentially a two-feature model.

### 10.4. Raw-Feature Permutation Importance (Locked Test)

![Raw Feature Permutation Importance](../outputs/04_best_model_and_feature_importance/raw_feature_permutation_importance.png)

Permutation importance measures the increase in MAE when a feature family's
values are randomly shuffled on the locked test set:

| Rank | Raw Feature | Mean Δ MAE (USD) | Std |
|------|-------------|----------------:|----:|
| 1 | `job_category` | 50,242 | 2,246 |
| 2 | `years_of_experience` | 12,796 | 600 |
| 3 | `country` | 5 | 22 |
| 4 | `company_size` | −27 | 60 |
| 5 | `education_required` | −41 | 63 |
| 6 | `skill_count` | −48 | 72 |
| 7 | `demand_score` | −56 | 176 |
| 8 | `city` | −65 | 187 |
| 9 | `benefits_score_10` | −83 | 106 |
| 10 | `remote_work` | −102 | 64 |
| 11 | `industry` | −103 | 125 |
| 12 | `job_title` | −122 | 83 |
| 13 | `required_skills` | −258 | 235 |

**Interpretation**: Only `job_category` and `years_of_experience` produce a
positive MAE increase when shuffled, confirming they are the only features
the model genuinely relies on for prediction accuracy.

### 10.5. Interpretation Caveats

The following caveats are **mandatory** for any reporting of these results:

1. **Two-feature concentration**: `job_category` and `years_of_experience`
   account for over **92%** of total feature importance. The model is
   essentially a two-feature model.

2. **Synthetic signal warning**: `years_of_experience` carries a
   contradictory, likely-synthetic signal (Stage 5/6 finding). Its importance
   MUST be reported as **model reliance**, not causal salary evidence.

3. **No causal claims**: Correlation and importance values are diagnostic
   association, not causal proof. No real-world salary-driver claim is
   permitted from this dataset.

4. **Honest R² framing**: The strong R² (0.803) is honestly reported as
   *"good fit to this specific dataset"*, never as *"AI salaries are
   explained by…"*.

### 10.6. Uncertainty Communication

Predictions MUST communicate uncertainty rather than bare point estimates:

| Metric | Value |
|--------|-------|
| Locked test rows | 298 |
| Mean prediction | ~197,787 USD |
| Median prediction | ~194,694 USD |
| 90th-percentile absolute error | 32,216 USD |
| **Practical prediction interval** | **± 32,216 USD** |

When presenting a salary prediction, the interval MUST be stated:
> "Predicted salary: $X ± $32,216 (90% of predictions fall within this band
> based on locked-test absolute errors)."

This interval reflects empirical prediction error, not a confidence interval
in the statistical sense. Its limitations:
- Based on a single locked-test period (March 2026, 298 rows)
- Assumes future data resembles the training distribution
- Does not account for systematic dataset limitations

---

## Source Artifact References

| Artifact | Path |
|----------|------|
| Run summary | `outputs/run_summary.json` |
| Model comparison (temporal CV) | `outputs/03_model_comparison/09_model_comparison_temporal_cv.csv` |
| Model comparison (fold metrics) | `outputs/03_model_comparison/09_model_comparison_fold_metrics.csv` |
| Best model tuning results | `outputs/04_best_model_and_feature_importance/10_best_model_tuning_results.csv` |
| Locked-test metrics | `outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv` |
| Encoded feature importance | `outputs/04_best_model_and_feature_importance/10_encoded_feature_importance.csv` |
| Raw permutation importance | `outputs/04_best_model_and_feature_importance/10_raw_feature_permutation_importance.csv` |
| Interpretation caveats | `outputs/04_best_model_and_feature_importance/10_interpretation_caveats.json` |
| Model metadata | `artifacts/metadata.json` |
| Model bundle | `artifacts/model_bundle.joblib` |
