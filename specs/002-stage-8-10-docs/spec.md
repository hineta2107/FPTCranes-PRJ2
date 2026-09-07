# Feature Specification: Stage 8–10 Model Evaluation Documentation

**Feature Branch**: `002-stage-8-10-docs`
**Created**: 2026-08-24
**Status**: Draft
**Input**: Write documentation (markdown only, no code) for model evaluation
stages 8–10, following the template structure in
`docs/Document_QD Project KHDL&AI.docx` and populated from the latest
pipeline output artifacts.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reviewer reads the Stage 8 Train–Test Split section (Priority: P1)

A project reviewer opens the documentation and reads the Stage 8 section to
verify that the temporal split is correctly described, the locked-test rule
is stated, the before/after data-processing comparison is present, and the
pipeline requirement (ColumnTransformer inside Pipeline) is documented.

**Why this priority**: The split definition is the foundation for all
downstream model evaluation claims. Without it, later sections lack context.

**Independent Test**: Open the markdown file, locate the Stage 8 heading,
and confirm it contains the split identity (2026-03 locked test), train/test
shapes (1,201 / 298), the locked-test rule, the before/after table, and the
failure-mode gate table.

**Acceptance Scenarios**:

1. **Given** the documentation file exists, **When** a reviewer reads
   Stage 8, **Then** it states the temporal split identity, partition sizes,
   and the locked-test rule verbatim.
2. **Given** the pipeline outputs exist, **When** the documentation is
   compared to `artifacts/metadata.json`, **Then** the train/test row
   counts match `development_rows` (1,201) and `locked_test.rows` (298).

---

### User Story 2 - Reviewer reads the Stage 9 Model Training & Comparison section (Priority: P1)

A reviewer reads Stage 9 to confirm that all candidate models are listed
with their temporal CV metrics, the Dummy baseline floor is included,
hyperparameter tuning rules are stated, and the SVM structural-mismatch
explanation is present.

**Why this priority**: The model comparison is the core evidence for model
selection and MUST be backed by the latest output artifacts.

**Independent Test**: Open the markdown, locate Stage 9, and verify the
comparison table matches `outputs/03_model_comparison/09_model_comparison_temporal_cv.csv`
exactly. Confirm the SVM explanation and the `years_of_experience` caveat
are present.

**Acceptance Scenarios**:

1. **Given** the documentation file exists, **When** a reviewer compares
   the model comparison table to the CSV, **Then** every model name, MAE,
   RMSE, R², and MedAE value matches.
2. **Given** the Dummy baseline is in the CSV, **When** a reviewer reads
   Stage 9, **Then** the Dummy row is present and all promoted models
   materially beat it.
3. **Given** the SVM under-performs, **When** a reviewer reads Stage 9,
   **Then** the structural-mismatch explanation (RBF distance in ~100-d
   sparse OHE space) is stated explicitly.

---

### User Story 3 - Reviewer reads the Stage 10 Best Model Selection & Feature Importance section (Priority: P1)

A reviewer reads Stage 10 to confirm the best model is documented with
locked-test metrics, both encoded and raw-family permutation importance
tables are present, interpretation caveats are stated, and uncertainty is
communicated.

**Why this priority**: This section contains the final model-selection
decision and the mandatory interpretation caveats required by the
constitution (Principle VI).

**Independent Test**: Open the markdown, locate Stage 10, and verify the
locked-test metrics match `outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv`,
the importance tables match the corresponding CSV files, and all four
interpretation caveats from `10_interpretation_caveats.json` are present.

**Acceptance Scenarios**:

1. **Given** the documentation file exists, **When** a reviewer compares
   the locked-test metrics to the CSV, **Then** MAE (14,961), RMSE
   (29,844), R² (0.803), MedAE (4,467) match.
2. **Given** the importance CSVs exist, **When** the documentation is
   compared, **Then** the top encoded features and raw-family permutation
   importance tables match the artifact values.
3. **Given** the interpretation caveats JSON exists, **When** a reviewer
   reads Stage 10, **Then** all four caveats (two-feature concentration,
   synthetic signal warning, no causal claims, honest R² framing) are
   stated verbatim or in equivalent wording.
4. **Given** the metadata contains `prediction_interval_abs_error_q90`,
   **When** a reviewer reads Stage 10, **Then** the ± 32,216 USD practical
   interval is communicated.

---

### Edge Cases

- What happens when a reviewer compares the documentation to a future
  pipeline re-run with different outputs? The documentation MUST state which
  run produced the numbers (date, seed, artifact paths).
- What happens if the best model changes in a future run? The documentation
  template structure MUST remain valid; only the values change.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The documentation MUST contain a Stage 8 section with the
  temporal split identity (2026-03 locked test), train shape (1,201 rows),
  test shape (298 rows), the locked-test rule, and a before/after
  data-processing comparison table.
- **FR-002**: The documentation MUST contain a Stage 9 section with a model
  comparison table listing all six candidates (Dummy, Linear Regression,
  Ridge, Random Forest, Gradient Boosting, SVR) with CV MAE, RMSE, R², and
  MedAE sourced from `09_model_comparison_temporal_cv.csv`.
- **FR-003**: Stage 9 MUST include the hyperparameter tuning rules
  (development folds only, RandomizedSearchCV, primary metric = negative
  MAE) and the tuning results for the winner from
  `10_best_model_tuning_results.csv`.
- **FR-004**: Stage 9 MUST explain SVR under-performance as a structural
  mismatch (RBF distance in sparse OHE space), not a tuning failure.
- **FR-005**: The documentation MUST contain a Stage 10 section stating the
  selected model (Random Forest), selection basis (lowest mean temporal CV
  MAE), and locked-test metrics (MAE, RMSE, R², MedAE) from
  `10_final_locked_test_metrics.csv`.
- **FR-006**: Stage 10 MUST include an encoded-feature importance table
  (top features from `10_encoded_feature_importance.csv`) and a raw-feature
  permutation importance table (from
  `10_raw_feature_permutation_importance.csv`).
- **FR-007**: Stage 10 MUST state all four interpretation caveats from
  `10_interpretation_caveats.json`: two-feature concentration (> 92%),
  synthetic signal warning for `years_of_experience`, no causal claims, and
  honest R² framing.
- **FR-008**: Stage 10 MUST communicate uncertainty via the 90th-percentile
  prediction interval (± 32,216 USD) rather than bare point estimates.
- **FR-009**: The documentation MUST follow the section structure and
  heading hierarchy of the template document
  (`docs/Document_QD Project KHDL&AI.docx`, stages 8–10).
- **FR-010**: All numeric values in the documentation MUST be traceable to
  the specific output artifact file paths listed in `outputs/run_summary.json`.
- **FR-011**: The documentation MUST be markdown only; no code, no notebook
  cells, no executable content.

### Constitutional Requirements *(mandatory)*

- **Data boundary**: N/A — this feature produces documentation only; no data
  processing or model training occurs.
- **Artifact contract**: The documentation reads from existing output
  artifacts (`outputs/03_model_comparison/`, `outputs/04_best_model_and_feature_importance/`,
  `artifacts/metadata.json`) and produces a single markdown file. No
  artifacts are modified.
- **Runtime boundary**: N/A — documentation is a static file; no runtime
  behavior.
- **Scientific interpretation**: The documentation MUST distinguish
  descriptive correlation, fitted-model importance, validation performance,
  and locked-test performance. No causal salary claims are permitted. The R²
  MUST be framed as "good fit to this specific dataset." The
  `years_of_experience` importance MUST be labeled as model reliance, not
  causal evidence.
- **Security and validation**: No untrusted inputs. No credentials. No
  runtime behavior.
- **Verification evidence**: The documentation is verified by comparing
  every numeric value to the source artifact CSV/JSON files. A reviewer
  checklist (cross-stage evaluation checklist from the analysis-requirement
  document) serves as acceptance evidence.

### Key Entities

- **Model Comparison Table**: Six-row table with model name, CV MAE mean/std,
  CV RMSE mean/std, CV R² mean/std, CV MedAE mean — sourced from the
  temporal CV CSV.
- **Locked-Test Metrics**: Single-row table with selected model, MAE, RMSE,
  R², MedAE, and selection basis.
- **Encoded Feature Importance**: Top-N table with encoded feature name and
  importance score.
- **Raw Feature Permutation Importance**: Table with raw feature family,
  mean Δ (USD), and std.
- **Interpretation Caveats**: Four mandatory statements about model
  limitations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every numeric value in the Stage 8–10 documentation matches
  the corresponding value in the source artifact file (zero discrepancies).
- **SC-002**: All four interpretation caveats from
  `10_interpretation_caveats.json` are present in the documentation.
- **SC-003**: The cross-stage evaluation checklist (Stage 8 split, Stage 9
  training, Stage 10 selection — 17 items from the analysis-requirement
  document) passes with 100% of items checked.
- **SC-004**: The documentation contains no code, no notebook cells, and no
  executable content.
- **SC-005**: A reviewer unfamiliar with the project can trace any displayed
  claim back to its source artifact within 60 seconds.

## Assumptions

- The pipeline has been run at least once and the latest outputs exist in
  `outputs/` and `artifacts/`.
- The template structure from `docs/Document_QD Project KHDL&AI.docx`
  (stages 8–10 headings and section order) is the authoritative format.
- The analysis-requirement document at
  `docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md`
  provides the acceptance checklist.
- No code changes are in scope; only markdown documentation is produced.
- The documentation will be placed under `docs/` in the project tree.
