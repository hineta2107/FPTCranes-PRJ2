# Implementation Plan: Model Training & Evaluation Pipeline

**Branch**: `001-model-training-evaluation` | **Date**: 2026-08-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-model-training-evaluation/spec.md`

## Summary

Validate and gap-fill Stages 8–10 of the AI Job Market Salary Prediction pipeline. The code already exists and runs end-to-end. Two gaps remain: (1) add DummyRegressor baseline to model catalog, (2) add interpretation caveats JSON output. All other work is validation tests proving existing behavior meets spec requirements. Phase-2 skills encoding (93-token multi-hot + `skill_count`) is already included.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: scikit-learn, pandas, numpy, joblib, rich (all already in `requirements.txt`)
**Storage**: File-based artifacts in `outputs/` and `artifacts/` directories
**Testing**: pytest (existing `tests/` directory)
**Target Platform**: Linux (local development + CI)
**Project Type**: Offline ML pipeline + Streamlit dashboard
**Performance Goals**: Full pipeline (Stages 8–10) completes in < 5 minutes on ~1,500 rows
**Constraints**: Deterministic results via `SEED=42`; locked test opened exactly once
**Scale/Scope**: 1,499 cleaned rows, 13 model features (11 structured + `required_skills` + `skill_count`), 189 encoded columns

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Data integrity**: Target = `annual_salary_usd`. Feature policy = `MODEL_FEATURES` (13 features including Phase-2 skills). Development period = all records before March 2026. Locked test = `posting_year=2026, posting_month=3`. Blocked columns = 12 fields in `BLOCKED_FEATURES`. All preprocessors (OrdinalEncoder, OneHotEncoder, CountVectorizer, StandardScaler, SimpleImputer) fitted on TRAIN partition only. ✅ PASS
- **Reproducibility**: `SEED=42` used in all stochastic operations. Input provenance recorded via `sha256_file()` in Stage 1. Each stage emits CSV evidence to `outputs/`. Artifact contract: `artifacts/model_bundle.joblib` + `artifacts/metadata.json`. ✅ PASS
- **Verification**: Tests verify: temporal split correctness, blocked column exclusion, preprocessor fitted on train only, all five models trained, metrics within expected ranges, bundle reload equivalence. Entrypoints: `python -m pipeline` and `pytest tests/`. ✅ PASS
- **Runtime boundary**: All training in `src/pipeline/` and `src/training/`. Streamlit (`streamlit.py`, `src/pages/`) loads artifacts read-only via `src/utils/artifacts.py`. No retraining in UI. ✅ PASS
- **Scientific and security claims**: Results are descriptive association on an academic dataset. No causal claims. `years_of_experience` synthetic-signal caveat required. Prediction intervals state empirical basis. User inputs validated against metadata schema. ✅ PASS
- **Simplicity**: No new infrastructure. All work uses existing scikit-learn Pipeline pattern and project module structure. ✅ PASS

## Project Structure

### Documentation (this feature)

```text
specs/001-model-training-evaluation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
src/
├── constants/
│   └── pipeline.py          # SEED, TARGET, MODEL_FEATURES, BLOCKED_FEATURES, etc.
├── models/
│   └── pipeline.py          # PipelinePaths, ModelDefinition, TrainingSelection
├── builder/
│   └── path_builder.py      # build_paths()
├── pipeline/
│   ├── __init__.py           # main() orchestrator (12 stages)
│   ├── train_test_split.py   # Stage 8
│   ├── model_training_comparison.py  # Stage 9
│   ├── best_model_selection_feature_importance_review.py  # Stage 10
│   └── save_deployable_pipeline_metadata.py  # Stage 11
├── training/
│   ├── __init__.py           # Public API re-exports
│   ├── build_preprocessor.py # ColumnTransformer construction
│   ├── build_model_pipeline.py  # Pipeline(preprocessor, model)
│   ├── build_pipeline.py     # Pipeline builder for best model
│   ├── model_catalog.py      # 5 ModelDefinition entries
│   ├── compare_models.py     # Temporal CV across catalog
│   ├── monthly_temporal_folds.py  # Expanding-window fold generator
│   ├── regression_metrics.py # MAE, RMSE, R², MedAE
│   ├── select_best_model.py  # Best model + tuning dispatch
│   ├── tune_gradient_boosting.py  # GB hyperparameter search
│   ├── tune_random_forest.py # RF hyperparameter search
│   ├── one_hot_encoder.py    # OneHotEncoder factory
│   └── out_of_fold_absolute_errors.py  # Interval estimation
└── utils/
    ├── artifacts.py          # load_csv, load_json, _load_model
    ├── pipeline_features.py  # normalize_skill_string, skill_count
    ├── pipeline_io.py        # save_json, sha256_file
    └── pipeline_plots.py     # plot_model_comparison, plot_best_outputs

tests/
└── test_pipeline_architecture.py  # Architecture, module, and pipeline tests
```

**Structure Decision**: The existing single-project structure with `src/pipeline/` (orchestration stages), `src/training/` (ML operations), and `src/models/` (dataclasses) is preserved. No structural changes — the architecture already maps 1:1 to Stages 8–10 requirements.

## Phases

### Phase 1: Gap-Filling (New Code)

**Goal**: Add the two missing pieces not yet in the codebase.

**Work items**:
1. Add `DummyRegressor(strategy="median")` to `src/training/model_catalog.py` as the non-ML floor baseline
2. Add `10_interpretation_caveats.json` output to `src/pipeline/best_model_selection_feature_importance_review.py` with 4 mandatory caveats + SVM structural-mismatch explanation
3. Re-run pipeline to regenerate outputs with DummyRegressor included

**Key files**: `src/training/model_catalog.py`, `src/pipeline/best_model_selection_feature_importance_review.py`

### Phase 2: Validation Tests (Verify Existing Behavior)

**Goal**: Add tests proving existing Stage 8–10 code meets all spec requirements.

**Existing code**: All Stage 8–10 modules already work. Tests validate, not implement.

**Work items**:
1. Test temporal split boundary (TRAIN=1,201 before March 2026, TEST=298 in March 2026)
2. Test all 12 `BLOCKED_FEATURES` absent from `MODEL_FEATURES`
3. Test `build_preprocessor()` includes CountVectorizer for Phase-2 skills
4. Test `model_catalog()` returns 6 models (after gap-fill)
5. Test `regression_metrics()` returns MAE, RMSE, R², MedAE
6. Test `monthly_temporal_folds()` produces expanding-window folds
7. Test `select_best_model()` returns valid TrainingSelection

**Key files**: `tests/test_stage8_10_requirements.py` (new)

### Phase 3: Output Verification Tests

**Goal**: Verify existing pipeline outputs match spec requirements by reading generated files.

**Work items**:
1. Verify `outputs/02_data_ready_for_machine_learning/08_split_summary.csv` has correct partition counts
2. Verify `outputs/03_model_comparison/09_model_comparison_temporal_cv.csv` has 6 model rows with all metrics
3. Verify `outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv` passes SC-003 thresholds (R² ≥ 0.75, MAE ≤ 20,000)
4. Verify encoded + permutation importance CSVs exist and show >90% concentration in top 2 features
5. Verify `10_interpretation_caveats.json` contains all 4 caveat keys (after gap-fill)
6. Verify `artifacts/model_bundle.joblib` reload equivalence and unknown-category handling
7. Verify `artifacts/metadata.json` contains all required fields including `interval_q90`

**Key files**: `tests/test_stage8_10_outputs.py` (new)

### Phase 4: Integration & Polish

**Goal**: End-to-end verification and graph update.

**Work items**:
1. Run full pipeline (`python pipeline.py`) and verify Stages 8–10 include DummyRegressor + caveats
2. Run `pytest tests/ -v` and verify all tests pass
3. Update `graphify` after code changes
4. Verify Streamlit loads the updated bundle correctly

## Complexity Tracking

No constitution violations. No complexity exceptions needed — all work uses existing patterns.

## Post-Design Constitution Re-Check

All five principles verified against the detailed phase design:
- Data integrity: Split, blocked features, train-only fit — all enforced by existing code paths ✅
- Reproducibility: SEED, provenance, CSV evidence — all present ✅
- Verification: Tests planned for each phase ✅
- Runtime boundary: No training in Streamlit ✅
- Scientific claims: Caveats required in Phase 3 output ✅
