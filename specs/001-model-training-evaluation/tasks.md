# Tasks: Model Training & Evaluation Pipeline

**Input**: Design documents from `specs/001-model-training-evaluation/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Context**: Stages 8–10 already have working code in `src/pipeline/` and `src/training/`.
The pipeline runs end-to-end and produces real outputs. Tasks below are scoped to:
1. **Gap-filling**: Add missing DummyRegressor baseline and interpretation caveats JSON
2. **Validation**: Add tests that verify existing behavior against spec requirements
3. **Verification**: Confirm existing outputs match expected values

**Tests**: Required by constitution. Each test names the behavior or contract it proves.

**Output Paths** (from `PipelinePaths` in `src/builder/path_builder.py`):
- `outputs/01_data_basic_clean/` → `paths.basic`
- `outputs/02_data_ready_for_machine_learning/` → `paths.ml_ready`
- `outputs/03_model_comparison/` → `paths.comparison`
- `outputs/04_best_model_and_feature_importance/` → `paths.best`
- `outputs/05_salary_prediction/` → `paths.prediction`
- `artifacts/` → `paths.artifacts`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)

---

## Phase 1: Gap-Filling (New Code)

**Purpose**: Add the two missing pieces not yet in the codebase

- [x] T001 Add `DummyRegressor(strategy="median")` to `src/training/model_catalog.py` — import from `sklearn.dummy`, add as first entry with `name="Dummy (Median)"`, `scale_numeric=False`
- [x] T002 Add interpretation caveats JSON output to `src/pipeline/best_model_selection_feature_importance_review.py` — save `10_interpretation_caveats.json` to `paths.best` containing 4 mandatory caveats: (1) two-feature concentration (`job_category` + `years_of_experience` > 92%), (2) `years_of_experience` synthetic-signal warning, (3) no causal claims permitted, (4) R² is "good fit to this dataset" not "salaries explained by…"
- [x] T003 [P] Add SVM structural-mismatch explanation to the caveats JSON from T002: "SVR under-performs due to RBF distance in ~100-d sparse OHE space, not a tuning bug"

### Already completed (logging gap-fill — FR-015):
- [x] T003a Added INFO/DEBUG logging to `src/pipeline/model_training_comparison.py` (Stage 9 orchestrator — was zero logging)
- [x] T003b Added DEBUG logging to `src/training/build_model_pipeline.py` (was zero logging)
- [x] T003c Enhanced logging in `src/pipeline/train_test_split.py` (added preprocessor fit, encoded feature count, shapes)
- [x] T003d Enhanced logging in `src/pipeline/best_model_selection_feature_importance_review.py` (added top-3 importance, interval value)
- [x] T003e Promoted `model_catalog.py` catalog log from DEBUG to INFO with candidate count

---

## Phase 2: Foundational Tests (Validate Existing Behavior)

**Purpose**: Tests proving existing Stage 8–10 code meets spec requirements. All tests validate already-working code.

- [x] T004 [US1] Write test: temporal split boundary — TRAIN contains only records before March 2026, TEST contains only `posting_year=2026, posting_month=3` — validate against actual split (TRAIN=1,201, TEST=298) in `tests/test_stage8_10_requirements.py`
- [x] T005 [P] [US1] Write test: all 12 `BLOCKED_FEATURES` are absent from `MODEL_FEATURES` constant in `tests/test_stage8_10_requirements.py`
- [x] T006 [P] [US1] Write test: `build_preprocessor()` includes a `CountVectorizer` transformer for `required_skills` (Phase-2 skills) in `tests/test_stage8_10_requirements.py`
- [x] T007 [P] [US1] Write test: preprocessor ColumnTransformer has 4 transformer steps (nominal, education, numeric, skills) in `tests/test_stage8_10_requirements.py`
- [x] T008 [US2] Write test: `model_catalog()` returns 6 models after T001 (Dummy, Linear, Ridge, RF, GB, SVR) in `tests/test_stage8_10_requirements.py`
- [x] T009 [P] [US2] Write test: `regression_metrics()` returns dict with keys `MAE`, `RMSE`, `R2`, `MedAE` in `tests/test_stage8_10_requirements.py`
- [x] T010 [P] [US2] Write test: `monthly_temporal_folds()` produces expanding-window folds where each fold's train indices are a strict subset of the next in `tests/test_stage8_10_requirements.py`
- [x] T011 [US3] Write test: `select_best_model()` returns a `TrainingSelection` dataclass with valid `model_name` and fitted `estimator` in `tests/test_stage8_10_requirements.py`
- [x] T012 [P] [US3] Write test: `out_of_fold_absolute_errors()` returns a numpy array with positive values and `np.quantile(errors, 0.90)` is positive in `tests/test_stage8_10_requirements.py`
- [x] T013 Run `pytest tests/test_stage8_10_requirements.py -v` and confirm all foundational tests pass

---

## Phase 3: Output Verification Tests (Validate Existing Artifacts)

**Purpose**: Tests that verify existing pipeline outputs match spec requirements. These read files already produced by the pipeline.

- [x] T014 [US1] Write test: `outputs/02_data_ready_for_machine_learning/08_split_summary.csv` exists and contains TRAIN_DEV=1,201 rows, LOCKED_TEST=298 rows in `tests/test_stage8_10_outputs.py`
- [x] T015 [P] [US1] Write test: `outputs/02_data_ready_for_machine_learning/08_encoded_feature_names.csv` exists and lists ~189 encoded features in `tests/test_stage8_10_outputs.py`
- [x] T016 [P] [US1] Write test: `outputs/02_data_ready_for_machine_learning/train_raw_model_input.csv` exists and none of the 12 blocked columns appear in its headers in `tests/test_stage8_10_outputs.py`
- [x] T017 [US2] Write test: `outputs/03_model_comparison/09_model_comparison_temporal_cv.csv` exists with 6 model rows (after T001) and columns `CV_MAE_mean`, `CV_RMSE_mean`, `CV_R2_mean`, `CV_MedAE_mean` in `tests/test_stage8_10_outputs.py`
- [x] T018 [P] [US2] Write test: at least one non-Dummy model has lower `CV_MAE_mean` than the Dummy baseline in comparison CSV in `tests/test_stage8_10_outputs.py`
- [x] T019 [P] [US2] Write test: `outputs/03_model_comparison/model_comparison_cv_r2.png` chart file exists in `tests/test_stage8_10_outputs.py`
- [x] T020 [US3] Write test: `outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv` exists with R² ≥ 0.75 and MAE ≤ 20,000 (SC-003 thresholds) in `tests/test_stage8_10_outputs.py`
- [x] T021 [P] [US3] Write test: `outputs/04_best_model_and_feature_importance/10_encoded_feature_importance.csv` exists, sorted descending, and top 2 features (`job_category` + `years_of_experience` families) account for >90% in `tests/test_stage8_10_outputs.py`
- [x] T022 [P] [US3] Write test: `outputs/04_best_model_and_feature_importance/10_raw_feature_permutation_importance.csv` exists with columns `raw_feature`, `importance_mean`, `importance_std` in `tests/test_stage8_10_outputs.py`
- [x] T023 [US3] Write test: `outputs/04_best_model_and_feature_importance/10_interpretation_caveats.json` exists (after T002) and contains all 4 required caveat keys in `tests/test_stage8_10_outputs.py`
- [x] T024 [US4] Write test: `artifacts/model_bundle.joblib` can be loaded, `.predict()` on a sample returns same shape as input rows in `tests/test_stage8_10_outputs.py`
- [x] T025 [P] [US4] Write test: `artifacts/metadata.json` contains keys `feature_columns`, `model_class`, `seed`, `content_hash`, `education_order`, `interval_q90` in `tests/test_stage8_10_outputs.py`
- [x] T026 [P] [US4] Write test: loaded pipeline handles unknown category via `handle_unknown="ignore"` — pass unseen `job_title` value without crash in `tests/test_stage8_10_outputs.py`
- [x] T027 [US5] Write test: `metadata.json` contains `interval_q90` > 0 in `tests/test_stage8_10_outputs.py`
- [x] T028 Run `pytest tests/test_stage8_10_outputs.py -v` and confirm all output verification tests pass

---

## Phase 4: Integration & Polish

**Purpose**: Re-run pipeline with DummyRegressor + caveats, verify everything, update graph

- [x] T029 Re-run full pipeline: `python pipeline.py` — verify Stages 8–10 now include DummyRegressor in comparison and produce `10_interpretation_caveats.json`
- [x] T030 Run full test suite: `pytest tests/ -v` and verify all tests pass (existing + new)
- [x] T031 [P] Run `graphify update .` to synchronize knowledge graph after code changes
- [x] T032 [P] Verify Streamlit dashboard loads updated bundle: confirm prediction page displays salary with uncertainty range

---

## Dependencies

```
Phase 1 (T001–T003: gap-filling) → Phase 4 (T029: re-run pipeline)
Phase 2 (T004–T013: unit tests) — can start immediately, independent of Phase 1
Phase 3 (T014–T028: output tests) — depends on Phase 4 for DummyRegressor + caveats rows
Phase 4 (T029–T032: integration) — depends on Phase 1
```

**Recommended execution order**:
1. T001–T003 (add DummyRegressor + caveats JSON) — new code
2. T004–T013 in parallel (unit tests on existing functions)
3. T029 (re-run pipeline to regenerate outputs with DummyRegressor)
4. T014–T028 (verify regenerated outputs)
5. T030–T032 (full suite + graph + Streamlit)

## Parallel Execution Opportunities

| Tasks | Why Parallel |
|-------|-------------|
| T005, T006, T007 | Independent assertions on different constants/preprocessor aspects |
| T009, T010 | Test different utilities (metrics vs folds) |
| T015, T016 | Independent file existence/content checks |
| T018, T019 | Independent comparison output checks |
| T021, T022 | Different importance type checks |
| T025, T026 | Different artifact property checks |
| T031, T032 | Graph update and Streamlit check independent |

## Implementation Strategy

**New code**: Only 3 tasks (T001–T003) write new code. Everything else validates existing behavior.
**MVP**: T001 + T029 = add DummyRegressor and re-run pipeline. Produces the core gap-fill.
**Full delivery**: All 32 tasks = complete validation suite proving all 14 FRs and 7 SCs.
