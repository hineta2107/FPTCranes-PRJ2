# Data Model: Model Training & Evaluation Pipeline

**Date**: 2026-08-23 | **Feature**: 001-model-training-evaluation

## Entities

### CleanedDataset

- **Source**: `data/ai_jobs_market_2025_2026.csv` → Stage 4 cleaning
- **Shape**: 1,499 rows × 25 columns
- **Target**: `annual_salary_usd` (int64, range 90,000–384,000 USD)
- **Temporal keys**: `posting_year`, `posting_month`

### TemporalPartition

| Partition | Filter | Expected Rows |
|-----------|--------|---------------|
| TRAIN/DEV | `posting_date < 2026-03` | ≈ 1,199–1,201 |
| LOCKED TEST | `posting_year=2026, posting_month=3` | ≈ 298–300 |

**Identity**: The split boundary (March 2026) is fixed; the ratio (~80/20) is a consequence.

### ModelFeatures (13 features)

| Feature | Type | Encoding |
|---------|------|----------|
| `job_title` | Nominal | OneHotEncoder (`handle_unknown="ignore"`) |
| `job_category` | Nominal | OneHotEncoder |
| `city` | Nominal | OneHotEncoder |
| `country` | Nominal | OneHotEncoder |
| `remote_work` | Nominal | OneHotEncoder |
| `company_size` | Nominal | OneHotEncoder |
| `industry` | Nominal | OneHotEncoder |
| `education_required` | Ordinal | OrdinalEncoder (5 levels) |
| `years_of_experience` | Numeric | Passthrough (+ optional StandardScaler) |
| `demand_score` | Numeric | Passthrough |
| `benefits_score_10` | Numeric | Passthrough |
| `skill_count` | Numeric (Phase-2) | Passthrough |
| `required_skills` | Text (Phase-2) | CountVectorizer (93-token TRAIN vocab, binary, `\|`-delimited) |

**Encoded shape**: ~189 columns after preprocessing.

### BlockedFeatures (12 columns — must not appear in X)

`job_id`, `salary_min_usd`, `salary_max_usd`, `salary_tier`, `experience_level`, `posting_year`, `posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`, `ai_salary_premium_pct`, `demand_growth_yoy_pct`

### ModelDefinition (dataclass)

- `name`: str — model display name
- `estimator_factory`: Callable → unfitted RegressorMixin
- `scale_numeric`: bool — whether to apply StandardScaler to numerics

**Instances**: DummyRegressor, Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, SVR (RBF)

### TrainingSelection (dataclass)

- `model_name`: str
- `estimator`: fitted RegressorMixin
- `scale_numeric`: bool
- `hyperparameters`: dict
- `tuning_results`: DataFrame

### ComparisonTable (DataFrame)

| Column | Type | Description |
|--------|------|-------------|
| `model` | str | Model name |
| `CV_MAE_mean` | float | Mean MAE across temporal folds |
| `CV_MAE_std` | float | Std of MAE |
| `CV_RMSE_mean` | float | Mean RMSE |
| `CV_RMSE_std` | float | Std of RMSE |
| `CV_R2_mean` | float | Mean R² |
| `CV_R2_std` | float | Std of R² |
| `CV_MedAE_mean` | float | Mean Median Absolute Error |

### InferenceBundle

- **File**: `artifacts/model_bundle.joblib` — serialized `Pipeline(preprocessor, model)`
- **Metadata**: `artifacts/metadata.json`
  - `feature_columns`: list of raw feature names
  - `model_class`: str
  - `training_date`: ISO timestamp
  - `seed`: int (42)
  - `content_hash`: SHA-256 of bundle file
  - `education_order`: list of ordinal levels
  - `interval_q90`: float (90th percentile half-width)

## Relationships

```
CleanedDataset --[temporal split]--> TemporalPartition (TRAIN, TEST)
TemporalPartition.TRAIN --[fit]--> Preprocessor (ColumnTransformer)
TemporalPartition.TRAIN --[train]--> ModelCandidate × 6
ModelCandidate × 6 --[compare]--> ComparisonTable
ComparisonTable --[select + tune]--> TrainingSelection
TrainingSelection --[fit on all dev]--> BestPipeline
BestPipeline --[predict]--> LockedTestPredictions
BestPipeline --[serialize]--> InferenceBundle
InferenceBundle --[loaded by]--> Streamlit (read-only)
```
