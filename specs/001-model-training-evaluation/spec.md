# Feature Specification: Model Training & Evaluation Pipeline

**Feature Branch**: `001-model-training-evaluation`
**Created**: 2026-08-23
**Status**: Draft
**Input**: Stages 8–10 analysis requirements — temporal train–test split, candidate model training & comparison, best model selection with feature importance review for the AI Job Market Salary Prediction project.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Temporal Train–Test Split (Priority: P1)

The data scientist executes the pipeline to split the cleaned dataset into a development partition and a locked test partition using a temporal boundary (March 2026), ensuring no future data leaks into the training process.

**Why this priority**: The temporal split is the foundation for all downstream model training, comparison, and evaluation. Without a correct split, all results are invalid.

**Independent Test**: Run the split step in isolation, verify partition shapes (TRAIN ≈ 1,199 rows, TEST ≈ 300 rows), confirm all 12 blocked columns are absent from predictors, and confirm the preprocessor is fitted only on the training partition.

**Acceptance Scenarios**:

1. **Given** the cleaned dataset with 1,499 rows and 25 columns, **When** the temporal split is executed with boundary `posting_year=2026, posting_month=3`, **Then** the TRAIN partition contains only records before March 2026 and the TEST partition contains only March 2026 records.
2. **Given** the split is complete, **When** the feature matrix X is inspected, **Then** none of the 12 blocked columns (`job_id`, `salary_min_usd`, `salary_max_usd`, `salary_tier`, `experience_level`, `posting_year`, `posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`, `ai_salary_premium_pct`, `demand_growth_yoy_pct`) appear.
3. **Given** the split is complete, **When** the preprocessor (ColumnTransformer) is inspected, **Then** it was fitted exclusively on TRAIN data, never on the full dataset or TEST partition.

---

### User Story 2 — Model Training & Comparison (Priority: P1)

The data scientist trains at least five required candidate models (DummyRegressor, Linear Regression, Ridge, Random Forest, Gradient Boosting) on the development partition using scikit-learn Pipelines, compares them using MAE, RMSE, and R², and identifies the best-performing candidate.

**Why this priority**: The core deliverable — comparing model candidates on identical splits with consistent metrics — is the primary output of Stages 9–10.

**Independent Test**: Train all five models, produce a sorted results table, and verify that all metrics are reported and the best candidate materially beats the DummyRegressor floor.

**Acceptance Scenarios**:

1. **Given** the TRAIN partition, **When** all five required models are trained inside Pipeline objects (ColumnTransformer + estimator), **Then** a comparison table is produced with MAE, RMSE, and R² for each model, sorted by R².
2. **Given** the comparison table, **When** the DummyRegressor baseline is examined, **Then** at least one candidate model materially beats it on both MAE and R².
3. **Given** hyperparameter tuning was performed, **When** the tuning history is inspected, **Then** tuning used only development folds (expanding monthly temporal folds), never the locked test partition.

---

### User Story 3 — Best Model Selection & Feature Importance (Priority: P2)

The data scientist selects the best model based on R² and MAE, runs the model on the locked test partition exactly once, reports feature importance (both encoded and raw-family permutation views), and produces an interpretation that includes all mandatory caveats.

**Why this priority**: Selection and interpretation provide the narrative and trust layer around model performance. Without caveats and importance analysis, the model results are misleading.

**Independent Test**: After model selection, open the locked test, produce evaluation metrics (MAE, RMSE, R², MedAE), generate feature importance charts, and verify all four mandatory interpretation caveats are present.

**Acceptance Scenarios**:

1. **Given** the best model is selected on development metrics, **When** it is evaluated on the locked test, **Then** MAE, RMSE, R², and MedAE are reported.
2. **Given** feature importance is computed, **When** encoded-feature importance and raw-family permutation importance are reviewed, **Then** the report explicitly states that `job_category` + `years_of_experience` account for >92% of total importance.
3. **Given** the final report is produced, **When** interpretation text is reviewed, **Then** all four caveats are present: (a) two-feature concentration, (b) `years_of_experience` synthetic-signal warning, (c) no causal claims, (d) R² described as "good fit to this dataset" not "AI salaries explained by…".

---

### User Story 4 — Artifact Hand-Off (Priority: P2)

The winning pipeline is serialized as a deployable inference bundle (preprocessor + model together) with metadata, ready for downstream consumption by the Streamlit application.

**Why this priority**: The artifact contract enables the downstream Streamlit predictor and is an explicit deliverable for Stage 11 hand-off.

**Independent Test**: Serialize the winning pipeline, reload it, pass a sample input, and verify the prediction is numerically identical to the in-memory pipeline output.

**Acceptance Scenarios**:

1. **Given** the best pipeline is trained, **When** it is serialized and reloaded, **Then** predictions on the same input are numerically identical.
2. **Given** the serialized bundle, **When** its metadata is inspected, **Then** it includes: feature schema, model class, training date, seed, and content hash.
3. **Given** the bundle is reloaded, **When** an unknown category is passed for a nominal feature, **Then** the pipeline handles it gracefully via `handle_unknown="ignore"` without crashing.

---

### User Story 5 — Uncertainty Communication (Priority: P3)

The final locked-test report communicates prediction uncertainty (e.g., practical interval ± 32,216 USD or MedAE) alongside point estimates, never presenting bare point predictions.

**Why this priority**: Honest uncertainty reporting is a constitutional requirement (Principle V) and prevents misinterpretation of model capability.

**Independent Test**: Review the final output and verify that every point prediction is accompanied by an uncertainty measure.

**Acceptance Scenarios**:

1. **Given** locked-test predictions are generated, **When** the summary is reviewed, **Then** it includes mean prediction, median prediction, and a 90% half-width interval or equivalent uncertainty measure.
2. **Given** the Streamlit dashboard consumes the model, **When** a user enters job parameters, **Then** the predicted salary is displayed with an uncertainty range, not a bare number.

---

### Edge Cases

- What happens when a new job category unseen in training is submitted at inference time? → Pipeline must handle via `handle_unknown="ignore"` and flag the input.
- What happens when `years_of_experience` is outside the observed training range? → Pipeline should predict if policy allows, but attach an out-of-distribution flag.
- What happens when the serialized bundle is reloaded and produces different predictions than the in-memory pipeline? → Release must fail; bundle must not be served.
- What happens if preprocessing is inadvertently fitted before the temporal split? → Run is invalidated; must rebuild from raw data.
- What happens if the locked test partition is used during tuning? → Final performance estimate is invalidated; must discard and retrain.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST perform a temporal train–test split using `posting_year=2026, posting_month=3` as the locked test boundary.
- **FR-002**: System MUST exclude all 12 blocked columns from the predictor feature set (`job_id`, `salary_min_usd`, `salary_max_usd`, `salary_tier`, `experience_level`, `posting_year`, `posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`, `ai_salary_premium_pct`, `demand_growth_yoy_pct`).
- **FR-003**: System MUST fit preprocessing (ColumnTransformer: OrdinalEncoder for `education_required`, OneHotEncoder with `handle_unknown="ignore"` for other nominals, passthrough for numerics) only on the TRAIN partition.
- **FR-004**: System MUST train at least five candidate models: DummyRegressor(strategy="median"), Linear Regression, Ridge (or ElasticNet), Random Forest, and Gradient Boosting (or HistGradientBoosting).
- **FR-005**: System MUST report MAE, RMSE, R², and MedAE for every candidate model.
- **FR-006**: System MUST perform hyperparameter tuning exclusively within development data using expanding monthly temporal folds, never touching the locked test.
- **FR-007**: System MUST select the best model based on highest R² and lowest MAE/RMSE, preferring simpler/more interpretable candidates when performance is similar.
- **FR-008**: System MUST produce encoded-feature importance from the estimator and raw-feature-family permutation importance on the locked test.
- **FR-009**: System MUST serialize the winning pipeline as a single deployable artifact (encoder + model together) with metadata (feature schema, seed, content hash).
- **FR-010**: System MUST communicate prediction uncertainty (interval width or MedAE) alongside point estimates in all outputs.
- **FR-011**: System MUST include all four mandatory interpretation caveats in the final report: two-feature concentration, synthetic-signal warning, no causal claims, and honest R² framing.
- **FR-012**: System MUST explain SVM under-performance as structural mismatch (RBF distance in ~100-d sparse OHE space), not hide or attribute it to tuning.
- **FR-013**: System MUST wrap each candidate model in a scikit-learn Pipeline whose first step is the ColumnTransformer preprocessor, ensuring the saved artifact is encoder + model together.
- **FR-014**: System MUST produce a comparison chart (horizontal bar, R² on test) for visual model comparison.
- **FR-015**: System MUST log every training step at INFO level (model name, stage transitions, results summary) and DEBUG level (fold shapes, metric values, parameters, encoded feature counts). Logging uses the project's Rich console handler via the standard `logging` module.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: Target is `annual_salary_usd`. The 11 KEEP features are the permitted predictor set. The temporal split boundary is March 2026. Blocked columns are documented in FR-002. All preprocessing (OrdinalEncoder, OneHotEncoder, scalers) MUST be fitted on the TRAIN partition only. The optional Phase-2 `required_skills` multi-hot encoding uses a TRAIN-only vocabulary (93 tokens).
- **Artifact contract**: The winning pipeline produces `artifacts/model_bundle.joblib` and `artifacts/metadata.json`. The bundle schema includes the ColumnTransformer preprocessor and estimator. Metadata includes feature_columns, model class, training date, seed, and content hash. The Streamlit application at `streamlit.py` is the primary consumer.
- **Runtime boundary**: All model fitting, tuning, and artifact generation occur in the offline pipeline (`pipeline.py` → `src/pipeline/` → `src/training/`). Streamlit loads artifacts read-only and never retrains.
- **Scientific interpretation**: Results are descriptive association and fitted-model performance on this specific academic dataset. No causal salary-driver claims permitted. `years_of_experience` carries a likely-synthetic signal. The R² is reported as "good fit to this dataset." Prediction intervals state their empirical basis.
- **Security and validation**: User inputs through Streamlit are validated against the feature schema recorded in metadata. Unknown categories are handled via `handle_unknown="ignore"`. Out-of-range numerics are flagged. Corrupt or missing artifacts produce actionable errors.
- **Verification evidence**: Tests must verify: temporal split correctness, blocked column exclusion, preprocessor-fitted-on-train-only, all five models trained, metrics match expected ranges, serialized bundle reload equivalence, and interpretation caveats present.

### Key Entities

- **CleanedDataset**: 1,499 rows × 25 columns from `data/ai_jobs_market_2025_2026.csv` after basic cleaning. Target: `annual_salary_usd`.
- **TemporalPartition**: TRAIN (pre-March 2026, ≈1,199 rows) and TEST (March 2026, ≈300 rows).
- **ModelCandidate**: One of five required estimators wrapped in a Pipeline with ColumnTransformer.
- **ComparisonTable**: DataFrame with model name, MAE, RMSE, R², MedAE, training time, sorted by R².
- **FeatureImportance**: Two views — encoded-feature importance (from estimator) and raw-family permutation importance (on locked test).
- **InferenceBundle**: Serialized pipeline artifact (`model_bundle.joblib`) + metadata JSON.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The temporal split produces a TRAIN partition with ≈1,199 rows and a TEST partition with ≈300 rows, with zero blocked columns in the feature matrix.
- **SC-002**: All five required candidate models are trained and produce a comparison table with MAE, RMSE, R², and MedAE.
- **SC-003**: The best candidate model achieves R² ≥ 0.75 and MAE ≤ 20,000 USD on the locked test, materially beating the DummyRegressor floor.
- **SC-004**: The serialized inference bundle, when reloaded, produces numerically identical predictions to the in-memory pipeline on the same input.
- **SC-005**: The feature importance report correctly identifies `job_category` + `years_of_experience` as accounting for >90% of total importance.
- **SC-006**: All four mandatory interpretation caveats are present in the final written report.
- **SC-007**: Every prediction output includes an uncertainty measure (interval or MedAE), not a bare point estimate.

## Clarifications

### Session 2026-08-23

- Q: Should Phase-2 skills encoding (required_skills → 93-token multi-hot + skill_count, 189 columns) be included as required or optional? → A: Include Phase-2 skills encoding as a required part of this feature.

## Assumptions

- The cleaned dataset at `data/ai_jobs_market_2025_2026.csv` is available and contains the expected 1,500 raw rows (1,499 after cleaning).
- The existing project architecture (`src/pipeline/`, `src/training/`, `src/models/`, `src/utils/`) is preserved — no structural changes to the project layout.
- scikit-learn is available in the project environment as the primary ML framework.
- Phase-2 skills enhancement (`required_skills` → 93-token TRAIN vocabulary multi-hot + `skill_count`, expanding to 189 encoded columns) is a required part of this feature.
- The downstream Streamlit consumer (`streamlit.py`) already has the loading and display logic for `model_bundle.joblib` and `metadata.json`.
- The `graphify` knowledge graph is used for code navigation and understanding during implementation.
