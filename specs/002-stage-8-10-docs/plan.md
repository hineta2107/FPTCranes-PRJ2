# Implementation Plan: Stage 8–10 Model Evaluation Documentation

**Branch**: `002-002-stage-8` | **Date**: 2026-08-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/002-stage-8-10-docs/spec.md`

## Summary

Produce a single markdown document covering stages 8–10 (Train–Test Split,
Model Training & Comparison, Best Model Selection & Feature Importance Review)
following the template structure from `docs/Document_QD Project KHDL&AI.docx`
and populated exclusively from the latest pipeline output artifacts.

## Technical Context

**Language/Version**: Markdown (no code)
**Primary Dependencies**: None — reads existing CSV/JSON artifacts
**Storage**: N/A — produces a static markdown file
**Testing**: Manual reviewer checklist against source artifacts
**Target Platform**: Documentation (renders in any markdown viewer)
**Project Type**: Documentation deliverable
**Performance Goals**: N/A
**Constraints**: All numeric values must match source artifacts exactly
**Scale/Scope**: Single markdown file, ~500–800 lines

## Constitution Check

*GATE: PASS — documentation-only feature with minimal constitutional surface.*

- **Data integrity**: N/A — no data processing. The documentation reads from
  existing artifacts but does not fit, transform, or split any data.
- **Reproducibility**: The documentation cites specific artifact file paths
  and the run that produced them (`outputs/run_summary.json`). Any future
  re-run produces new artifacts that would require a documentation update.
- **Verification**: Acceptance is verified by comparing every numeric value in
  the documentation to the source CSV/JSON files. The cross-stage evaluation
  checklist (17 items) from the analysis-requirement document serves as the
  test. No pytest or pipeline execution is required.
- **Runtime boundary**: N/A — the documentation is a static file with no
  runtime behavior.
- **Scientific and security claims**: The documentation MUST state all four
  interpretation caveats (two-feature concentration, synthetic signal,
  no causal claims, honest R² framing). No causal salary claims permitted.
  No secrets or untrusted inputs involved.
- **Simplicity**: No exceptions required. A single markdown file is the
  simplest possible deliverable.

## Project Structure

### Documentation (this feature)

```text
specs/002-stage-8-10-docs/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: source artifact inventory
├── data-model.md        # Phase 1: document entity/table schemas
├── quickstart.md        # Phase 1: how to regenerate/update the docs
└── tasks.md             # Phase 2 output (created by /speckit-tasks)
```

### Deliverable Location

```text
docs/
└── stage_08_10_model_evaluation.md   # The final documentation file
```

### Source Artifacts (read-only inputs)

```text
outputs/
├── 03_model_comparison/
│   ├── 09_model_comparison_temporal_cv.csv
│   └── 09_model_comparison_fold_metrics.csv
├── 04_best_model_and_feature_importance/
│   ├── 10_final_locked_test_metrics.csv
│   ├── 10_encoded_feature_importance.csv
│   ├── 10_raw_feature_permutation_importance.csv
│   ├── 10_best_model_tuning_results.csv
│   └── 10_interpretation_caveats.json
└── run_summary.json

artifacts/
├── metadata.json
└── model_bundle.joblib
```

**Structure Decision**: Single markdown file under `docs/` reading from
existing output artifacts. No code, no tests, no new infrastructure.

## Phases

### Phase 0: Research — Artifact Inventory

No technical unknowns to resolve. The "research" is an inventory of the
source artifacts and their schemas:

| Artifact | Schema/Columns | Purpose in documentation |
|----------|---------------|--------------------------|
| `09_model_comparison_temporal_cv.csv` | model, CV_MAE_mean/std, CV_RMSE_mean/std, CV_R2_mean/std, CV_MedAE_mean | Stage 9 comparison table |
| `10_final_locked_test_metrics.csv` | selected_model, MAE, RMSE, R2, MedAE, selection_basis | Stage 10 selection summary |
| `10_encoded_feature_importance.csv` | encoded_feature, importance | Stage 10 encoded importance table |
| `10_raw_feature_permutation_importance.csv` | raw_feature, importance_mean, importance_std | Stage 10 raw importance table |
| `10_best_model_tuning_results.csv` | candidate_id, n_estimators, min_samples_leaf, max_features, max_depth, CV_MAE_mean/std | Stage 9 tuning results |
| `10_interpretation_caveats.json` | 5 keys: two_feature_concentration, synthetic_signal_warning, no_causal_claims, honest_r2_framing, svm_structural_mismatch | Stage 9–10 caveats |
| `artifacts/metadata.json` | model features, locked_test, development_rows, numeric_ranges, prediction_interval_abs_error_q90 | Stage 8 split info + Stage 10 interval |

### Phase 1: Document Structure Design

The markdown document follows the template heading hierarchy:

```markdown
# C. DESIGN PROCESS (continued)

## STAGE 8. Train-Test Split
  - Temporal split identity and rationale
  - Train/test shapes table
  - Locked-test rule
  - Before vs After Data Processing table
  - Failure-mode gate table

## STAGE 9. Model Training & Comparison
  - Candidate model ladder table
  - Hyperparameter tuning rules
  - Tuning results (winner)
  - Model comparison table (temporal CV metrics)
  - Evaluation metrics definitions
  - Model promotion criteria
  - Narrative: SVM explanation, years_of_experience caveat

## STAGE 10. Best Model Selection & Feature Importance Review
  - Selected model + selection basis
  - Locked-test metrics table
  - Encoded feature importance table (top 10)
  - Raw-feature permutation importance table
  - Interpretation caveats (all 4)
  - Uncertainty communication (±32,216 USD interval)
```

### Phase 2: Writing Tasks

Defined by `/speckit-tasks` — not created here.

## Complexity Tracking

No violations. No complexity exceptions required.
