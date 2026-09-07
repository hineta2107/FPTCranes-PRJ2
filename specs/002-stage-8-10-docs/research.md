# Research: Stage 8–10 Documentation Artifact Inventory

**Date**: 2026-08-24
**Status**: Complete — no unknowns to resolve

## Decision Log

### D1: Source of truth for numeric values

**Decision**: All numbers come from CSV/JSON artifacts in `outputs/` and
`artifacts/`, never from memory or the docx template.

**Rationale**: Constitution Principle VI.8 requires artifact traceability.
The template document contains reference numbers from an earlier run that
may differ from the current temporal-protocol outputs.

**Alternatives considered**: Using the docx template numbers directly —
rejected because they reflect a random-split protocol (Gradient Boosting
winner, R² = 0.852), not the current temporal-protocol outputs (Random
Forest winner, R² = 0.803).

### D2: Document format and location

**Decision**: Single markdown file at `docs/stage_08_10_model_evaluation.md`.

**Rationale**: Matches the existing `docs/analysis-requirement/` pattern.
A single file keeps stages 8–10 together as a coherent evaluation narrative.

**Alternatives considered**: Three separate files per stage — rejected as
over-splitting for a ~600-line document.

### D3: Heading hierarchy

**Decision**: Follow the template docx structure: `## STAGE N. Title` for
each stage, with subsections as `### N.X. Subtitle`.

**Rationale**: Maintains consistency with the project template and allows
the document to be inserted into the larger project report.

**Alternatives considered**: Using `#` for stages — rejected to maintain
compatibility with the overall document hierarchy where `#` is reserved
for the top-level section (`C. DESIGN PROCESS`).

## Artifact Schema Summary

| File | Key columns/fields | Values (latest run) |
|------|-------------------|---------------------|
| `metadata.json` → `development_rows` | int | 1,201 |
| `metadata.json` → `locked_test.rows` | int | 298 |
| `metadata.json` → `locked_test.year/month` | int | 2026 / 3 |
| `metadata.json` → `model_name` | str | "Random Forest" |
| `metadata.json` → `prediction_interval_abs_error_q90` | float | 32,216.37 |
| `09_model_comparison_temporal_cv.csv` | 6 rows × 7 cols | See plan §Phase 0 |
| `10_final_locked_test_metrics.csv` | 1 row × 6 cols | MAE=14,961; R²=0.803 |
| `10_encoded_feature_importance.csv` | ~189 rows × 2 cols | Top: job_category_AI Engineering = 0.591 |
| `10_raw_feature_permutation_importance.csv` | 13 rows × 3 cols | Top: job_category = 50,242 |
| `10_best_model_tuning_results.csv` | 4 rows × 7 cols | Winner: n_estimators=400, min_samples_leaf=2 |
| `10_interpretation_caveats.json` | 5 keys | All four mandatory caveats present |
