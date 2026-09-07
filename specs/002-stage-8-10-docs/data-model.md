# Data Model: Stage 8–10 Documentation Entities

**Date**: 2026-08-24

## Entities

These are the logical tables/structures that appear in the documentation.

### ModelComparisonRow

Represents one row in the Stage 9 model comparison table.

| Field | Type | Source |
|-------|------|--------|
| model | string | CSV column |
| CV_MAE_mean | float (USD) | CSV column |
| CV_MAE_std | float (USD) | CSV column |
| CV_RMSE_mean | float (USD) | CSV column |
| CV_RMSE_std | float (USD) | CSV column |
| CV_R2_mean | float (0–1) | CSV column |
| CV_R2_std | float | CSV column |
| CV_MedAE_mean | float (USD) | CSV column |

### LockedTestMetrics

Single-row summary of the best model's locked-test performance.

| Field | Type | Source |
|-------|------|--------|
| selected_model | string | CSV column |
| MAE | float (USD) | CSV column |
| RMSE | float (USD) | CSV column |
| R2 | float (0–1) | CSV column |
| MedAE | float (USD) | CSV column |
| selection_basis | string | CSV column |

### EncodedFeatureImportance

One row per encoded feature (pipeline column name).

| Field | Type | Source |
|-------|------|--------|
| encoded_feature | string | CSV column |
| importance | float (0–1) | CSV column |

### RawFeaturePermutationImportance

One row per raw feature family.

| Field | Type | Source |
|-------|------|--------|
| raw_feature | string | CSV column |
| importance_mean | float (USD Δ) | CSV column |
| importance_std | float (USD) | CSV column |

### TuningCandidate

One row per hyperparameter trial for the winner model.

| Field | Type | Source |
|-------|------|--------|
| candidate_id | int | CSV column |
| n_estimators | int | CSV column |
| min_samples_leaf | int | CSV column |
| max_features | float | CSV column |
| max_depth | float/null | CSV column |
| CV_MAE_mean | float (USD) | CSV column |
| CV_MAE_std | float (USD) | CSV column |

### InterpretationCaveats

Key-value pairs of mandatory interpretation warnings.

| Key | Value type | Required? |
|-----|-----------|-----------|
| two_feature_concentration | string | YES |
| synthetic_signal_warning | string | YES |
| no_causal_claims | string | YES |
| honest_r2_framing | string | YES |
| svm_structural_mismatch | string | YES |
