# Quickstart: Model Training & Evaluation Pipeline

## Prerequisites

- Python 3.11+
- Dependencies installed: `pip install -r requirements.txt`
- Dataset present: `data/ai_jobs_market_2025_2026.csv`

## Run the Full Pipeline

```bash
cd /home/mihuynh-ubuntu/minhhuynh/FPTCranes/FPTCranes-PRJ2
python pipeline.py
```

This executes all 12 stages. Stages 8–10 (this feature) produce:

| Stage | Output Directory | Key Files |
|-------|-----------------|-----------|
| 8 | `outputs/ml_ready/` | `08_split_summary.csv`, `08_encoded_feature_names.csv`, `train_raw_model_input.csv` |
| 9 | `outputs/comparison/` | `09_model_comparison_temporal_cv.csv`, `09_model_comparison_fold_metrics.csv` |
| 10 | `outputs/best/` | `10_final_locked_test_metrics.csv`, `10_encoded_feature_importance.csv`, `10_raw_feature_permutation_importance.csv` |
| 11 | `artifacts/` | `model_bundle.joblib`, `metadata.json` |

## Run Tests

```bash
pytest tests/ -v
```

## Run Streamlit Dashboard

```bash
streamlit run streamlit.py
```

The dashboard loads `artifacts/model_bundle.joblib` and `artifacts/metadata.json` read-only.

## Key Constants

All defined in `src/constants/pipeline.py`:

- `SEED = 42`
- `TARGET = "annual_salary_usd"`
- `LOCKED_YEAR = 2026`, `LOCKED_MONTH = 3`
- `MODEL_FEATURES` = 13 features (11 structured + `required_skills` + `skill_count`)
- `BLOCKED_FEATURES` = 12 columns excluded from predictors
