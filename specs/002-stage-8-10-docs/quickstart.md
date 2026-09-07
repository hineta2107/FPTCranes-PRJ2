# Quickstart: Stage 8–10 Model Evaluation Documentation

## Prerequisites

- Pipeline has been run at least once (`outputs/run_summary.json` exists
  with `"status": "SUCCESS"`)
- All output artifacts exist in `outputs/03_model_comparison/` and
  `outputs/04_best_model_and_feature_importance/`

## How to produce the documentation

This is a **documentation-only** deliverable. No code execution is needed.

1. Read the source artifacts listed in the plan's Phase 0 artifact inventory.
2. Write `docs/stage_08_10_model_evaluation.md` following the heading
   structure defined in the plan's Phase 1.
3. Copy numeric values exactly from the CSV/JSON files — do not round or
   reformat unless the plan specifies a display format.

## How to verify

Run through the 17-item cross-stage evaluation checklist in
`docs/analysis-requirement/stage_08_10_model_training_and_evaluation.md`
§5 ("Cross-Stage Evaluation Checklist"). Every box must be checkable.

## How to update after a pipeline re-run

1. Re-run the pipeline: `python -m src.pipeline`
2. Compare `outputs/run_summary.json` timestamp to the documentation's
   stated run.
3. If outputs differ, update all numeric values in
   `docs/stage_08_10_model_evaluation.md` from the new artifact files.
