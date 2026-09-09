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

## Lưu ý quan trọng về dữ liệu đầu vào

### Tương thích định dạng cột

Pipeline hỗ trợ **hai định dạng** file CSV:

| Định dạng | Cột danh mục | Mô tả |
|-----------|-------------|--------|
| **Chuẩn** | `job_category` | File CSV có header đúng — pipeline xử lý bình thường |
| **Corrupt** | `AI Engineering` | File CSV bị lỗi header (giá trị rò rỉ lên tên cột) — pipeline tự động phát hiện, loại bỏ dòng corrupt và đổi tên cột về `job_category` |

### Sử dụng file dữ liệu tùy chỉnh

```bash
python pipeline.py --data path/to/your_file.csv
```

**Các cột bắt buộc:**

| Cột | Mô tả |
|-----|--------|
| `annual_salary_usd` | Biến mục tiêu (target) — lương hàng năm (USD) |
| `job_id` | Mã định danh duy nhất cho mỗi tin tuyển dụng |
| `job_category` hoặc `AI Engineering` | Danh mục công việc (chấp nhận một trong hai tên cột) |
| `posting_year` | Năm đăng tin |
| `posting_month` | Tháng đăng tin |
| `required_skills` | Kỹ năng yêu cầu, phân tách bởi dấu `\|` |

> **Lưu ý:** Nếu file CSV của bạn có cột `AI Engineering` thay vì `job_category`, pipeline sẽ tự động xử lý — phát hiện các dòng bị corrupt (giá trị `"job_category"` xuất hiện trong cột dữ liệu), loại bỏ chúng, và đổi tên cột về `job_category` trước khi tiến hành các bước phân tích tiếp theo.

## Local Streamlit demo login

The app supports two types of accounts:

| Account | Username | Password     | Access level |
|---------|----------|--------------|--------------|
| Admin   | `admin`  | `AIJob2026!` | Full access — all pages, market evaluation fields (Demand score, Benefits score) visible |
| User    | `user`   | `user123`    | Candidate view — market evaluation fields hidden, set automatically to medians |

Override admin credentials before shared deployment with environment variables:

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

## Recent Updates (Changelog)

- **UI Refactoring:** Reorganized `_model_comparison.py`, `_best_model.py`, and `_prediction.py` to use a clean 4-tab layout for better UX.
- **Bug Fix:** Fixed `NameError: name 'px' is not defined` in `_prediction.py` by adding the missing `plotly.express` import.
- **Chart Restoration:** Restored the missing Runtime Performance charts in `_model_comparison.py`, displaying Fit time and Predict time side-by-side.
- **Dynamic Study Cases:** Changed the Study Cases section in `_prediction.py` to dynamically load random real records from the raw dataset (`ai_jobs_market_2025_2026.csv`) instead of using hardcoded mock data. Added a button to randomly redraw the cases.
- **Data Resilience:** Enhanced `_load_raw_data()` in `_prediction.py` to automatically handle the corrupted CSV header (`AI Engineering` -> `job_category`) and dynamically calculate `skill_count` based on `required_skills` during inference, preventing `KeyError` during prediction.
