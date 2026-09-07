# Tasks: Stage 8–10 Model Evaluation Documentation

**Input**: Design documents from `specs/002-stage-8-10-docs/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Verification is by reviewer checklist comparison against source
artifacts. No pytest or code tests — this is a documentation-only deliverable.

**Organization**: Tasks are grouped by user story (one per stage) to enable
independent writing and verification.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different sections, no dependencies)
- **[Story]**: Which user story this task belongs to (US1=Stage 8, US2=Stage 9, US3=Stage 10)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create the documentation file and establish structure

- [x] T001 Create `docs/stage_08_10_model_evaluation.md` with the top-level heading and section skeleton (Stage 8, 9, 10 headings per plan Phase 1 structure)
- [x] T002 Add provenance header noting the pipeline run date, artifact paths, and run_summary status from `outputs/run_summary.json`

**Checkpoint**: Empty skeleton file exists with correct heading hierarchy.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extract and format all source data that multiple sections reference

- [x] T003 Read `artifacts/metadata.json` and extract: development_rows (1,201), locked_test.rows (298), locked_test.year (2026), locked_test.month (3), model_name ("Random Forest"), model_features (13 features), blocked_features (12 blocked), prediction_interval_abs_error_q90 (32,216.37)
- [x] T004 Read `outputs/03_model_comparison/09_model_comparison_temporal_cv.csv` and format as a markdown table (6 models × 7 metric columns)
- [x] T005 [P] Read `outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv` and format as a markdown table
- [x] T006 [P] Read `outputs/04_best_model_and_feature_importance/10_encoded_feature_importance.csv` and format the top 10 rows as a markdown table
- [x] T007 [P] Read `outputs/04_best_model_and_feature_importance/10_raw_feature_permutation_importance.csv` and format all 13 rows as a markdown table
- [x] T008 [P] Read `outputs/04_best_model_and_feature_importance/10_best_model_tuning_results.csv` and format as a markdown table (4 candidates)
- [x] T009 [P] Read `outputs/04_best_model_and_feature_importance/10_interpretation_caveats.json` and extract all 5 caveat statements

**Checkpoint**: All source data extracted and formatted as markdown tables ready for insertion.

---

## Phase 3: User Story 1 — Stage 8 Train–Test Split (Priority: P1)

**Goal**: Write the complete Stage 8 section documenting the temporal split,
locked-test rule, before/after comparison, and failure-mode gate.

**Independent Test**: Compare Stage 8 section to the 5-item checklist in
`docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md` §5.1.

- [x] T010 [US1] Write the temporal split subsection: state split identity (2026-03 locked test), train shape (1,201 rows × 13 features), test shape (298 rows), and the ~80/20 ratio note in `docs/stage_08_10_model_evaluation.md`
- [x] T011 [US1] Write the locked-test rule subsection: 2026-03 never in CV, labels not inspected until all decisions frozen, inadvertent use invalidates the run in `docs/stage_08_10_model_evaluation.md`
- [x] T012 [US1] Write the pipeline requirement subsection: split on raw cleaned df, each model wrapped in Pipeline with ColumnTransformer preprocessor (OrdinalEncoder + OneHotEncoder + passthrough) in `docs/stage_08_10_model_evaluation.md`
- [x] T013 [P] [US1] Write the Before vs After Data Processing table (6-row comparison: data structure, corrupted values, experience signals, encoding, target leakage, model readiness) in `docs/stage_08_10_model_evaluation.md`
- [x] T014 [P] [US1] Write the Failure-Mode Gate table (5 rows: preprocessing fitted before split, locked test used during tuning, unknown category, out-of-range numeric, bundle reload) in `docs/stage_08_10_model_evaluation.md`

**Checkpoint**: Stage 8 section complete. All 5 items in §5.1 checklist pass.

---

## Phase 4: User Story 2 — Stage 9 Model Training & Comparison (Priority: P1)

**Goal**: Write the complete Stage 9 section documenting the candidate ladder,
tuning rules, comparison results, and narrative explanations.

**Independent Test**: Compare Stage 9 section to the 6-item checklist in
`docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md` §5.2.

- [x] T015 [US2] Write the candidate model ladder subsection listing all 6 models (Dummy, Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR) with role and rationale in `docs/stage_08_10_model_evaluation.md`
- [x] T016 [US2] Write the hyperparameter tuning rules subsection (5 rules: dev folds only, RandomizedSearchCV, primary metric = negative MAE, logging, simplicity preference) in `docs/stage_08_10_model_evaluation.md`
- [x] T017 [US2] Insert the tuning results table (from T008) showing the 4 Random Forest candidates with selected hyperparameters (n_estimators=400, min_samples_leaf=2, max_features=0.8) in `docs/stage_08_10_model_evaluation.md`
- [x] T018 [US2] Insert the model comparison table (from T004) with all 6 models ranked by CV MAE in `docs/stage_08_10_model_evaluation.md`
- [x] T019 [US2] Write the evaluation metrics definitions table (MAE, RMSE, R², MedAE with formula, priority, and interpretation) in `docs/stage_08_10_model_evaluation.md`
- [x] T020 [US2] Write the SVM structural-mismatch explanation paragraph: RBF distance in ~100-d sparse OHE space causes near-Dummy performance, not a tuning bug in `docs/stage_08_10_model_evaluation.md`
- [x] T021 [US2] Write the years_of_experience caveat paragraph: contradictory/likely-synthetic signal from Stage 5/6, check importance before trusting R² in `docs/stage_08_10_model_evaluation.md`

**Checkpoint**: Stage 9 section complete. All 6 items in §5.2 checklist pass.

---

## Phase 5: User Story 3 — Stage 10 Best Model Selection & Feature Importance (Priority: P1)

**Goal**: Write the complete Stage 10 section documenting the selection
decision, importance tables, interpretation caveats, and uncertainty.

**Independent Test**: Compare Stage 10 section to the 6-item checklist in
`docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md` §5.3.

- [x] T022 [US3] Write the model selection subsection: Random Forest selected by lowest mean temporal CV MAE on development period; locked test opened once after selection/tuning frozen in `docs/stage_08_10_model_evaluation.md`
- [x] T023 [US3] Insert the locked-test metrics table (from T005): MAE=14,961, RMSE=29,844, R²=0.803, MedAE=4,467 in `docs/stage_08_10_model_evaluation.md`
- [x] T024 [US3] Insert the encoded feature importance table (top 10 from T006) with note that job_category_AI Engineering (0.591) + years_of_experience (0.275) = 86.6% in `docs/stage_08_10_model_evaluation.md`
- [x] T025 [US3] Insert the raw-feature permutation importance table (from T007) with note that job_category (50,242 USD) + years_of_experience (12,796 USD) dominate in `docs/stage_08_10_model_evaluation.md`
- [x] T026 [US3] Write the four interpretation caveats (from T009): two-feature concentration >92%, synthetic signal warning, no causal claims, honest R² framing in `docs/stage_08_10_model_evaluation.md`
- [x] T027 [US3] Write the uncertainty communication subsection: 90th-percentile prediction interval ± 32,216 USD, median prediction ~194,694 USD, mean ~197,787 USD in `docs/stage_08_10_model_evaluation.md`

**Checkpoint**: Stage 10 section complete. All 6 items in §5.3 checklist pass.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final review and cross-stage consistency

- [x] T028 [P] Add artifact source references footer listing all CSV/JSON file paths used
- [x] T029 [P] Verify all numeric values against source artifacts (spot-check 5 values per stage)
- [x] T030 Run the full 17-item cross-stage evaluation checklist from `docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md` §5 and confirm all items pass
- [x] T031 [P] Run `graphify update .` to keep the knowledge graph current

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (file must exist)
- **User Stories (Phases 3–5)**: All depend on Phase 2 (tables formatted)
  - US1, US2, US3 can proceed **in parallel** (different sections)
- **Polish (Phase 6)**: Depends on all user stories being complete

### Within Each User Story

- Insert pre-formatted tables from Phase 2
- Write narrative around them
- Verify against the per-stage checklist

### Parallel Opportunities

- T005, T006, T007, T008, T009 can all run in parallel (Phase 2)
- T013, T014 can run in parallel (Phase 3)
- All three user story phases (3, 4, 5) can run in parallel
- T028, T029, T031 can run in parallel (Phase 6)

---

## Implementation Strategy

### MVP First (Stage 8 Only)

1. Complete Phase 1: Setup (skeleton)
2. Complete Phase 2: Extract all data
3. Complete Phase 3: Write Stage 8
4. **VALIDATE**: Check §5.1 checklist
5. Deliver Stage 8 as first increment

### Full Delivery

1. Setup + Foundational → all data extracted
2. Write all three stages in parallel → verify each checklist
3. Polish → full 17-item checklist passes
4. Done — single markdown file deliverable

---

## Notes

- This is a documentation-only feature: no code, no tests, no runtime
- All tasks write to a single file: `docs/stage_08_10_model_evaluation.md`
- Numeric values MUST be copied exactly from artifacts, never approximated
- The 17-item cross-stage checklist is the final acceptance gate
