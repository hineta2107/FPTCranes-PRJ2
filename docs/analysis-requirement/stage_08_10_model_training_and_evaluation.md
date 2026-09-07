# Analysis Requirement — Stages 8–10: Model Training, Comparison & Evaluation

> **Scope of this document.** This file consolidates the *analysis requirements* for the
> model training/comparison/evaluation section of the AI Job Market Salary
> Prediction project. It covers **Stage 8 (Train–Test Split)**, **Stage 9 (Model
> Training & Comparison)** and **Stage 10 (Best Model Selection & Feature
> Importance Review)** as defined in
> `docs/Document_QD Project KHDL&AI/document_qd_project_khdl_ai_part02.md`
> (lines 106–220) and elaborated in `part03.md` / `part04.md`.
>
> **Purpose.** Serve as the acceptance-criteria and evaluation reference used to
> judge the modelling section of the project against the source documentation
> and against the raw dataset `data/ai_jobs_market_2025_2026.csv`.
>
> **No code is included in this file.** All snippets already live in the source
> notebook / `part03.md`; this document is a *requirement & evaluation
> specification* only.

---

## 1. Context & Input Contract

### 1.1 Dataset under analysis
- **File:** `data/ai_jobs_market_2025_2026.csv`
- **Raw shape:** 1,500 rows × 25 columns.
- **After basic cleaning** (corrupted `AI Engineering = "job_category"` row
  removed, canonical rename to `job_category`): **1,499 rows**.
- **Target:** `annual_salary_usd` (int64, observed range 90,000 – 384,000 USD).
- **Temporal columns:** `posting_year ∈ {2025, 2026}`, `posting_month ∈ 1..12`.

### 1.2 Feature contract entering Stage 8
Stages 8–10 receive the **11 KEEP features** produced by Stages 5–7:

| # | Feature | Role | Type |
|---|---------|------|------|
| 1 | `job_title` | Nominal | KEEP |
| 2 | `job_category` (formerly `AI Engineering`) | Nominal | KEEP |
| 3 | `years_of_experience` | Numeric | KEEP (ablation-sensitive) |
| 4 | `education_required` | Ordinal | KEEP |
| 5 | `city` | Nominal | KEEP |
| 6 | `country` | Nominal | KEEP |
| 7 | `remote_work` | Nominal | KEEP |
| 8 | `company_size` | Nominal | KEEP |
| 9 | `industry` | Nominal | KEEP |
| 10 | `demand_score` | Numeric | KEEP |
| 11 | `benefits_score_10` | Numeric | KEEP |

Optionally the Phase-2 skill enhancement adds:
`required_skills` → multi-hot encoded (93-token TRAIN vocabulary) + engineered
`skill_count`, taking the encoded feature count to **189 columns**.

**Blocked from X** (must not appear as predictors): `job_id`, `salary_min_usd`,
`salary_max_usd`, `salary_tier`, `experience_level`, `posting_year`,
`posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`,
`ai_salary_premium_pct`, `demand_growth_yoy_pct`.

---

## 2. Stage 8 — Train–Test Split

### 2.1 Requirement
The split **must be temporal, not random**. The locked test period is
`posting_year = 2026 AND posting_month = 3` (March 2026), everything strictly
before it is TRAIN/DEV.

### 2.2 Expected shapes
| Partition | Filter | Rows | Columns |
|-----------|--------|------|---------|
| TRAIN / DEV | posting date < 2026-03-01 | ~1,199 (≈ 1,201 in Phase-2) | 11 raw / 189 encoded |
| Locked test | posting_year = 2026, posting_month = 3 | ~300 (≈ 298 in Phase-2) | same schema as TRAIN |

The split ratio is therefore ≈ **80 / 20**, but the *identity* of the split
(March 2026) is what matters, not the ratio.

### 2.3 Locked-test rule
- The 2026-03 partition is **never** used inside cross-validation.
- Its labels **must not** be inspected until *all* of the following are frozen:
  feature set, model class, hyperparameters, interval policy, and the narrative
  template used to report results.
- Any inadvertent use of test data during selection/tuning invalidates the
  final performance estimate; the run must be discarded and rebuilt.

### 2.4 Pipeline requirement
- Splitting is performed on the **raw cleaned `df`**, *not* on the
  `pd.get_dummies` correlation view (`df_corr`).
- Each candidate model is wrapped in a single scikit-learn `Pipeline` whose
  first step is a `ColumnTransformer` preprocessor
  (OrdinalEncoder for `education_required`, OneHotEncoder with
  `handle_unknown="ignore"` for the other nominals, passthrough for numerics).
  The saved artifact is therefore **encoder + model together**.

### 2.5 Before / after processing (evaluation checklist)

| Aspect | Before | After (required state before Stage 9) |
|--------|--------|---------------------------------------|
| Data structure | 25 raw columns incl. IDs, leakage, free text | 11 modelling features, leakage-free |
| Corrupted values | Header value leaked into a data row | Row removed, field renamed |
| Experience signals | Two contradictory columns | Redundant column dropped |
| Encoding | Unsafe `get_dummies` (analysis only) | Deployable `ColumnTransformer` |
| Target leakage | Present (`salary_min/max/tier`) | Fully eliminated |
| Model readiness | ❌ Not ready | ✅ Ready — single deployable artifact |

### 2.6 Failure-mode gate

| Failure mode | Required remediation |
|--------------|----------------------|
| Preprocessing fitted before split | Invalidate run and rebuild from raw data |
| Locked test used during tuning | Invalidate final performance estimate |
| Unknown category at inference | Encode safely, flag, monitor rate |
| Out-of-range numeric input | Predict only if policy allows; attach OOD flag |
| Bundle reload changes prediction | Release fails; do not serve |

---

## 3. Stage 9 — Model Training & Comparison

### 3.1 Required candidate ladder
The submission **must** compare, on the identical train/test split, at least:

| # | Model | Role | Rationale |
|---|-------|------|-----------|
| 0 | `DummyRegressor(strategy="median")` | Non-ML floor | Any promoted model must beat it materially |
| 1 | Linear Regression | Transparent linear baseline | Detect additive relationships |
| 2 | Ridge (or ElasticNet) | Regularized linear baseline | Stable under many one-hot columns |
| 3 | Random Forest | Nonlinear bagging baseline | Robust tabular comparator |
| 4 | Gradient Boosting (or HistGradientBoosting) | Primary advanced candidate | Strong nonlinear tabular baseline |
| 5 | SVR (RBF kernel) | Optional diagnostic | Expected to underperform on ~100-d sparse OHE |

Optional extensions (Extra Trees, CatBoost, LightGBM, XGBoost) are permitted
**only** if the dependency policy allows them and they use the same splits and
metrics.

### 3.2 Hyperparameter tuning rules
1. Tuning is nested inside the **development** data only, using expanding
   monthly temporal folds.
2. Use `RandomizedSearchCV` or a small bounded grid — the dev set has only
   ~1.2 k rows.
3. Optimize primary metric = **negative MAE** (weighted multi-metric is
   allowed) while also recording RMSE, R², and fold variance.
4. Log for every trial: parameters, feature-set ID, split/fold IDs, seed,
   train time, inference time, metrics.
5. Prefer the **simpler, more stable, more interpretable** candidate when two
   models are statistically/practically similar.

### 3.3 Evaluation metrics (all must be reported)

| Metric | Formula | Priority | Interpretation |
|--------|---------|----------|----------------|
| MAE | (1/n) · Σ |yᵢ − ŷᵢ| | **Primary** | Average absolute USD error |
| RMSE | √((1/n) · Σ (yᵢ − ŷᵢ)²) | Primary-secondary | Penalizes large misses |
| R² | 1 − SSres / SStot | Diagnostic | Variance explained (careful on synthetic data) |
| MedAE | median(|yᵢ − ŷᵢ|) | Robust diagnostic | Typical error, outlier-robust |
| Interval coverage | mean(yᵢ ∈ [Lᵢ, Uᵢ]) | If intervals used | Empirical coverage |
| Interval width | mean(Uᵢ − Lᵢ) | If intervals used | Interpret together with coverage |

### 3.4 Model promotion gates (all must be GO)

| Gate | GO condition |
|------|--------------|
| Predictive value | MAE/RMSE materially beat Dummy, stable across temporal folds |
| Temporal generalization | Locked-test degradation is understood and acceptable |
| Feature plausibility | No blocked fields; suspicious feature-family gains documented via ablation |
| Error slices | No severe unexplained degradation in role / domain / location slices |
| Complexity | Inference time, artifact size, maintenance burden justified |
| Explainability | Global + local explanations available; no causal claims |
| Reproducibility | Reproducible from fingerprint + config + env + seed |
| Deployment equivalence | Reloaded bundle numerically equivalent to offline pipeline |

### 3.5 Reference results (baseline `train_test_split`, `random_state=42`)

From `part02.md` §9.3 — used as the **expected reference** the submission is
evaluated against:

| Rank | Model | MAE (USD) | RMSE (USD) | R² |
|------|-------|-----------|------------|-----|
| 1 | Gradient Boosting | 14,522.78 | 25,808.91 | **0.852** |
| 2 | Random Forest | 14,742.45 | 27,639.93 | 0.830 |
| 3 | Ridge Regression | 25,347.58 | 35,734.20 | 0.716 |
| 4 | Linear Regression | 25,393.36 | 35,754.04 | 0.716 |
| 5 | SVM (RBF kernel) | 47,989.42 | 64,351.12 | 0.080 |

From `part04.md` §F.4 — reference under the **temporal, locked-test** protocol
(Phase-2, skills included):

| Model | MAE | RMSE | R² | MedAE |
|-------|-----|------|-----|-------|
| Random Forest (winner under temporal protocol) | 14,961 | 29,844 | **0.803** | 4,467 |

### 3.6 Required narrative points
The write-up **must** explicitly address:
- Why the SVM lands near / below the dummy floor — **structural mismatch**
  (RBF distance in ~100-d sparse OHE space), not a tuning bug.
- The `years_of_experience` caveat carried from the correlation step: its
  strong negative relationship with salary is a *possible synthetic-data
  artifact*.
- The rule "check feature importance before trusting the R²" — the R² alone
  must not be reported without §4 diagnostics.

---

## 4. Stage 10 — Best Model Selection & Feature Importance Review

### 4.1 Selection rule
- Best model = highest R² **and** lowest MAE/RMSE on the evaluation split.
- If two candidates are close (fold-variance-overlapping), pick the **simpler
  and more interpretable** one.
- The **decision must be frozen** before the locked-test labels are opened.

### 4.2 Expected best models

| Protocol | Winner | Test R² | Test MAE |
|----------|--------|---------|----------|
| Simple 80/20 random split (part02) | Gradient Boosting | 0.852 | 14,523 USD |
| Temporal locked-test protocol (part04) | Random Forest | 0.803 | 14,961 USD |

Both are acceptable outcomes; the *interpretation section* must match the
protocol actually used.

### 4.3 Feature-importance requirements

**4.3.1 Encoded-feature importance (from the estimator itself)** — reference
values, Gradient Boosting reference run:

| Encoded feature | Importance |
|-----------------|-----------|
| `AI Engineering_AI Engineering` (core domain) | 0.7237 |
| `years_of_experience` | 0.2056 |
| `AI Engineering_Security` | 0.0119 |
| `AI Engineering_Robotics` | 0.0092 |
| `AI Engineering_Architecture` | 0.0076 |
| all remaining features | < 1 % each |

Random Forest reference (temporal, part04):

| Encoded feature | Importance |
|-----------------|-----------|
| `nominal__job_category_AI Engineering` | 0.5908 |
| `numeric__years_of_experience` | 0.2753 |
| `nominal__job_category_Robotics` | 0.0136 |
| `nominal__job_category_Security` | 0.0091 |
| `numeric__demand_score` | 0.0054 |
| others | < 1 % each |

**4.3.2 Raw-feature-family permutation importance** (on the locked test) —
required for the temporal protocol:

| Raw feature family | Mean Δ (USD) | Std |
|--------------------|-------------:|----:|
| `job_category` | 50,242 | 2,246 |
| `years_of_experience` | 12,796 | 600 |
| `country` | 5 | 22 |
| `company_size` | −27 | 60 |
| `education_required` | −41 | 63 |
| `skill_count` | −48 | 72 |
| `demand_score` | −56 | 176 |
| `city` | −65 | 187 |

### 4.4 Mandatory interpretation caveats
The submission is judged on whether it states **explicitly** that:
1. `job_category` + `years_of_experience` account for **> 92 %** of total
   feature importance — the model is essentially a two-feature model.
2. Because `years_of_experience` carries a contradictory / likely-synthetic
   signal (Stage 6 finding), its importance **must be reported as *model
   reliance*, not causal salary evidence**.
3. Correlation and importance values are **diagnostic association**, not
   causal proof — no real-world salary-driver claim is permitted.
4. The strong R² is honestly reported as *"good fit to this specific
   dataset"*, never as *"AI salaries are explained by…"*.

### 4.5 Locked-test summary — reference numbers (part04, §F.5)

| Locked test rows | Mean prediction | Median prediction | 90 % half-width interval |
|-----------------:|----------------:|------------------:|-------------------------:|
| 298 | 197,787 USD | 194,694 USD | ± 32,216 USD |

The final report **must** communicate uncertainty (e.g. the ±32,216 USD
practical interval) rather than presenting a bare point estimate.

---

## 5. Cross-Stage Evaluation Checklist

Use this checklist to score the Stage 8–10 deliverable against the source
documentation and the raw CSV.

### 5.1 Stage 8 — Split
- [ ] Split is **temporal** (2026-03 as locked test), not a random 80/20.
- [ ] TRAIN ≈ 1,199 rows, TEST ≈ 300 rows (± Phase-2 delta).
- [ ] Split performed on **raw cleaned `df`**, not on `df_corr`.
- [ ] All 12 BLOCKED columns absent from X.
- [ ] Preprocessing fitted **inside** the pipeline, not before the split.

### 5.2 Stage 9 — Training & comparison
- [ ] All five required models trained on identical splits.
- [ ] `DummyRegressor` floor is reported.
- [ ] MAE, RMSE, R² reported for every model.
- [ ] Tuning nested in development folds; locked test never touched.
- [ ] SVM under-performance is *explained*, not hidden.
- [ ] Winning candidate materially beats Dummy on both MAE and R².

### 5.3 Stage 10 — Selection & importance
- [ ] Best model documented with test MAE, RMSE, R², MedAE.
- [ ] Both encoded importance and raw-family permutation importance reported
      (or the reason one is omitted is documented).
- [ ] `job_category` + `years_of_experience` concentration is called out.
- [ ] `years_of_experience` synthetic-signal caveat is repeated.
- [ ] Uncertainty is communicated (interval or MedAE), not just a point.
- [ ] No causal claim made about real-world salary drivers.

### 5.4 Traceability to source docs
| Requirement source | Location |
|--------------------|----------|
| Split rule, locked test, before/after table | `part02.md` §Stage 8 (lines 106–128) |
| Candidate ladder, tuning, metrics, gates | `part02.md` §Stage 9 (lines 129–196) |
| Best model + top importances | `part02.md` §Stage 10 (lines 197–212) |
| Pipeline & feature-importance code reference | `part03.md` (Model Selection / Save Artifact) |
| Temporal-protocol reference numbers & permutation table | `part04.md` §F.3–F.5 |
| Deployable-artifact contract used downstream | `part04.md` §D.2 |

---

## 6. Deliverables Expected From Stage 8–10 Owner

The Stage 8–10 owner (per the `H — TASK ASSIGNMENT TABLE`, "Model Training &
Comparison, Best Model Selection and Feature Importance Review" is Huỳnh Minh)
must produce:

1. A reproducible notebook cell block that:
   - Executes the temporal split.
   - Trains the five required models inside `Pipeline`s.
   - Prints a `results_df` sorted by R².
2. A comparison chart (horizontal bar, R² on test) — required in `part02.md`
   and `part03.md`.
3. A locked-test evaluation table matching the schema in §3.5 / §4.5.
4. A feature-importance section covering both encoded and raw-family views.
5. A short written interpretation covering every point in §4.4.
6. Hand-off of the winning pipeline object to Stage 11 (Hiển) for artifact
   serialization (`preprocessor.pkl`, `model.pkl`, `feature_columns.json`,
   `metadata.json`).

---

*End of Stage 8–10 analysis-requirement specification.*
