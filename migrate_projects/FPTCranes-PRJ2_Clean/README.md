# FPTCranes-PRJ2 Optimized — AI Job Market Salary Prediction

Production-style academic project for **AI Job Market Salary Prediction** using a 12-stage, data-quality-first and leakage-safe workflow.

This optimized package keeps the requested Streamlit presentation structure while incorporating the stronger output/evidence coverage used in `Project_HKII_G3`.

## Streamlit menu

1. Data basic clean — Stages 01–05
2. Data ready for ML — Stages 06–08
3. Model comparison — Stage 09
4. Best model — Stages 10–11
5. Salary prediction — Stage 12
6. 12-stage pipeline

## Scientific / ML controls

- Locked temporal test: **March 2026**
- Target-aware diagnostics use **DEV only** before final-test opening
- TRAIN/DEV-only preprocessing and 93-token skill vocabulary
- Explicit feature leakage policy and ablation plan
- Pearson + Spearman diagnostics, numeric VIF, pairwise redundancy audit
- Expanding-window temporal CV
- Dummy baseline + five regression families
- Bounded model tuning
- Locked-test MAE / RMSE / R² / MedAE
- Encoded importance + raw-family permutation importance
- Bundle reload-equivalence deployment gate
- Streamlit inference uses the same serialized model bundle

## Current verified run

- Raw rows: 1,500
- Clean rows: 1,499
- Development rows: 1,201
- Locked-test rows: 298
- Selected model: Random Forest
- Locked-test MAE: ~15,054 USD
- Locked-test RMSE: ~29,993 USD
- Locked-test R²: ~0.801
- Locked-test MedAE: ~4,440 USD
- Automated tests: 13/13 passed
- Generated analytical charts: 27

## Project structure

```text
FPTCranes-PRJ2_Optimized/
├── config/
│   └── project.yaml
├── data/raw/
│   └── ai_jobs_market_2025_2026.csv
├── src/
│   ├── builder/
│   ├── components/
│   ├── constants/
│   ├── pages/
│   ├── pipeline/
│   ├── training/
│   └── utils/
├── outputs/
│   ├── 01_data_basic_clean/
│   ├── 02_data_ready_for_machine_learning/
│   ├── 03_model_comparison/
│   ├── 04_best_model_and_feature_importance/
│   └── 05_salary_prediction/
├── artifacts/
├── assets/
├── reports/
├── tests/
├── pipeline.py
├── pineline.py
├── streamlit.py
└── requirements.txt
```

## Run

```bash
python -m pip install -r requirements.txt
python pipeline.py
python -m pytest -q
streamlit run streamlit.py
```

Backward-compatible alias:

```bash
python pineline.py
```

## Local Streamlit demo login

Default academic demo credentials are in `config/project.yaml`:

- username: `admin`
- password: `AIJob2026!`

Override them before shared deployment with environment variables:

```bash
AI_JOB_USER=your_user
AI_JOB_PASSWORD=your_password
```

## Main output charts shown in Streamlit

### Data basic clean
- Stage 02 target distribution
- Stage 03 missingness and high-cardinality review
- Stage 05 logic issue rates
- Experience-level / years-of-experience contradiction charts
- Salary by job category
- Salary range integrity

### Data ready for ML
- Feature governance mix
- Top target correlations
- Pearson vs Spearman
- Numeric VIF
- Top 30 skills
- Temporal split timeline and split donut

### Model comparison
- CV MAE
- CV R²
- Fold-by-fold MAE stability
- Feature-family ablation
- Feature-importance drift

### Best model
- Actual vs predicted
- Residual distribution
- Raw feature-family permutation importance
- Encoded feature importance

### Salary prediction
- Interactive serving form
- Empirical prediction interval
- OOD / review flags
- Locked-test serving evidence charts
