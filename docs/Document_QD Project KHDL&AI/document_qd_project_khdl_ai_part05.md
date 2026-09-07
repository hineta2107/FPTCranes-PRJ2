*Part 5 of 5*
# G - REFERENCES
eBook
eBook
eBook
eBook

# H - TASK ASSIGNMENT TABLE

| TASK ASSIGNMENT TABLE | TASK ASSIGNMENT TABLE | TASK ASSIGNMENT TABLE | TASK ASSIGNMENT TABLE |
| --- | --- | --- | --- |
| No. | Main task | Student name | Completion - % |
| 1 | Load data & Data Quality Check | Gia Bảo |  |
| 2 | Contradictory-Feature Investigation, Feature Selection & Correlation Analysis | Nguyễn Kim Ngân |  |
| 3 | Encoding, Scaling, Feature Engineer, and Leakage Prevention | Duy AI |  |
| 4 | Model Training & Comparison, Best Model Selection and Feature Importance Review | Huỳnh Minh |  |
| 5 | Streamlit Salary Prediction Dashboard & Visualization | Tạ Gia Hiển |  |

| STAGE | NGƯỜI PHỤ TRÁCH | GHI CHÚ |
| --- | --- | --- |
| 1. Load Data | Bảo | Sau bước này có data cơ bản clean |
| 2. Project Scope & Initial Inspection | Bảo | Sau bước này có data cơ bản clean |
| 3. Data Quality Check | Bảo | Sau bước này có data cơ bản clean |
| 4. Corrupted Row Removal | Bảo | Sau bước này có data cơ bản clean |
| 5. Contradictory-Feature Investigation | Ngân | Sau bước này có data ready cho train model |
| 6. Feature Selection & Leakage Preventaion | Ngân & Duy AI | Sau bước này có data ready cho train model |
| 7. Correlation Encoding & Analysis | Duy AI | Sau bước này có data ready cho train model |
| 8. Train-Test Split | Minh | Sau bước này chọn được model và tính năng quan trọng |
| 9. Model Training & Comparison | Minh | Sau bước này chọn được model và tính năng quan trọng |
| 10. Best Model Selection & Feature Importance Review | Minh | Sau bước này chọn được model và tính năng quan trọng |
| 11. Save Deployable Pipeline + Metadata | Hiển | Đây là module dựng dự đoán lương cho tương lai khi nhập thông tin vào feature X |
| 12. Streamlit Salary Prediction Dashboard | Hiển | Đây là module dựng dự đoán lương cho tương lai khi nhập thông tin vào feature X |

- Data Dictionary and Modeling Role – before cleaning:

| Source column | dtype | Observed domain | Role | Policy | Technical note |
| --- | --- | --- | --- | --- | --- |
| job_id | object | 1500 unique | ID | BLOCK | Identifier; suffix is strongly ordered with salary. |
| job_title | object | 25 unique | Nominal | KEEP | Core role feature; 25 categories. |
| job_category (old name = AI Engineering) | object | 13 unique | Nominal | KEEP after cleaning | Canonical rename to job_category; remove invalid row value "job_category". |
| experience_level | object | 4 unique | Ordinal/category | ABLATION | Semantically inconsistent with years_of_experience; do not trust blindly. |
| years_of_experience | int64 | 1 .. 15; 15 unique | Numeric | ABLATION | Strong predictive signal but implausible downward salary relationship; production promotion requires provenance. |
| education_required | object | 5 unique | Ordinal | KEEP | Explicit ordered encoder; unknown categories -> -1 or dedicated Unknown. |
| annual_salary_usd | int64 | 90000 .. 384000; 248 unique | Target | TARGET | Regression target only. |
| salary_min_usd | int64 | 90000 .. 180000; 17 unique | Target-adjacent | BLOCK primary | Fixed by job_title and annual salary lies outside stated range in 40.3% of rows. |
| salary_max_usd | int64 | 180000 .. 320000; 16 unique | Target-adjacent | BLOCK primary | Fixed by job_title; logically inconsistent with target for many rows. |
| city | object | 20 unique | Nominal | KEEP/ABLATE | 20 values; deterministically maps to country in this snapshot. |
| country | object | 14 unique | Nominal | KEEP/ABLATE | 14 values; redundant with city for current snapshot. |
| remote_work | object | 3 unique | Nominal | KEEP | 3 categories. |
| company_size | object | 5 unique | Nominal | KEEP | 5 categories. |
| industry | object | 12 unique | Nominal | KEEP | 12 categories. |
| required_skills | object | 1500 unique | Multi-label text | PHASE 2 | Raw string is unique in all 1,500 rows; tokenize by | into train-only vocabulary (93 observed tokens). |
| ai_salary_premium_pct | float64 | 3 .. 18; 151 unique | Numeric | CONDITIONAL | Use only if provenance confirms it is available before target and not computed from compensation. |
| demand_score | int64 | 68 .. 98; 20 unique | Numeric | KEEP/ABLATE | 20 values; exact function of job_title in this snapshot, so it adds no independent information if title is present. |
| demand_growth_yoy_pct | float64 | 5 .. 87.8; 565 unique | Numeric | CONDITIONAL | Availability/provenance gate; excluded from conservative baseline. |
| benefits_score_10 | int64 | 6 .. 10; 5 unique | Numeric | KEEP | Observed range 6-10. |
| posting_year | int64 | 2025 .. 2026; 2 unique | Temporal | SPLIT / CONDITIONAL X | Required for temporal split; use as feature only if deployment date is known and ablation supports it. |
| posting_month | int64 | 1 .. 12; 12 unique | Temporal | SPLIT / CONDITIONAL X | Same as posting_year; avoid full-dataset temporal preprocessing. |
| is_senior | int64 | 0 .. 1; 2 unique | Derived flag | BLOCK primary | Exact deterministic derivative of experience_level in supplied data. |
| is_remote_friendly | int64 | 0 .. 1; 2 unique | Derived flag | BLOCK primary | Exact deterministic derivative of remote_work in supplied data. |
| is_llm_role | int64 | 0 .. 1; 2 unique | Derived flag | BLOCK primary | Deterministic by job_title in supplied data; redundant if job_title is present. |
| salary_tier | object | 5 unique | Target-adjacent | BLOCK primary | Only 27.5% consistent with annual_salary_usd band; not safe as predictor or evaluation label. |
