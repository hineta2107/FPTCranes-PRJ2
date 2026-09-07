# Research: Model Training & Evaluation Pipeline

**Date**: 2026-08-23 | **Feature**: 001-model-training-evaluation

## Research Summary

No NEEDS CLARIFICATION items exist — the technical context is fully resolved from the existing codebase and analysis requirements document.

## Decisions

### 1. DummyRegressor Baseline

**Decision**: Add `DummyRegressor(strategy="median")` to `model_catalog()` as the non-ML floor.
**Rationale**: The analysis requirement (§3.1) mandates a dummy baseline to prove predictive value. The existing catalog has 5 models (Linear, Ridge, RF, GB, SVR) but no DummyRegressor.
**Alternatives considered**: `DummyRegressor(strategy="mean")` — rejected because median is more robust to salary distribution skew.

### 2. Phase-2 Skills Encoding

**Decision**: Include `required_skills` multi-hot encoding and `skill_count` as required model features.
**Rationale**: Clarified during `/speckit-clarify` — user chose to include Phase-2 skills. The existing `build_preprocessor()` already has a `CountVectorizer` transformer for `required_skills` and `MODEL_FEATURES` already includes both `required_skills` and `skill_count`.
**Alternatives considered**: Baseline-only (11 features) — rejected per user decision.

### 3. Temporal Split Protocol

**Decision**: Use the existing temporal split (`LOCKED_YEAR=2026`, `LOCKED_MONTH=3`) already coded in `src/constants/pipeline.py`.
**Rationale**: The split is already implemented in `src/pipeline/train_test_split.py` and matches the analysis requirement (§2.1).
**Alternatives considered**: Random 80/20 split — rejected per analysis requirement mandate.

### 4. Hyperparameter Tuning Strategy

**Decision**: Use existing `tune_gradient_boosting()` and `tune_random_forest()` with `RandomizedSearchCV` on development folds.
**Rationale**: Both tuning modules already exist in `src/training/`. They use `monthly_temporal_folds()` for expanding-window CV, matching §3.2 requirements.
**Alternatives considered**: Grid search — rejected because dev set has only ~1.2k rows; randomized search is more efficient.

### 5. Interpretation Caveats Output

**Decision**: Add a `10_interpretation_caveats.json` output file in Stage 10 containing the four mandatory caveats as structured data.
**Rationale**: The caveats must be verifiable by tests (Constitution Principle III). A JSON file enables automated validation.
**Alternatives considered**: Embed caveats only in narrative markdown — rejected because it cannot be machine-tested.
