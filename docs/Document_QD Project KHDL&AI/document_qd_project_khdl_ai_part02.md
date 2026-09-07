*Part 2 of 5*
## **STAGE 7. Correlation Encoding & Analysis**
(PIC = Duy AI)
- All preprocessing lives inside a scikit-learn Pipeline/ColumnTransformer serialized with the model.
**7.1. Imputation and Transformation:**
- Imputation (Điền khuyết): Thực hiện tính toán (ví dụ: Median) trên tập Train và áp dụng giá trị đó lên tập Test. Với lương, nên dùng Median thay vì Mean vì lương thường có phân phối lệch (skewed).
- Log Transformation: Dữ liệu lương thường bị lệch phải (nhiều người lương thấp, ít người lương cực cao). Áp dụng log1p sẽ giúp phân phối chuẩn hơn, giúp các mô hình tuyến tính hoạt động hiệu quả hơn.
**7.2. Correlation analysis & VIF - Variance Inflation Factor (Phân tích tương quan & Đa cộng tuyến) on Train:**
- Correlation analysis (Phân tích tương quan): Sử dụng Pearson hoặc Spearman để xem các yếu tố như "Level", "Skill" tương quan thế nào với "Salary".
- VIF (Variance Inflation Factor): Kiểm tra hiện tượng đa cộng tuyến giữa các biến độc lập.
**7.3. Encoding **(Mã hóa):
- Nominal categoricals: OneHotEncoder job_title, AI Engineering, city, country, remote_work, company_size, industry — with handle_unknown="ignore" so unseen categories at prediction time don't crash the app; use min_frequency only if supported by validation and documented.
- Education: OrdinalEncoder with explicit order Bootcamp/Self-taught < Associate's < Bachelor's < Master's < PhD. Unknown values map to a dedicated fallback code.

- Numeric values (years_of_experience, demand_score, benefits_score_10): median imputation for inference robustness even though current training snapshot has no missing values; fit median on TRAIN only.
- Skills (Phase 2): split on |, trim, de-duplicate tokens, build vocabulary on TRAIN, optionally MultiLabelBinarizer / token TF-IDF. Raw required_skills string is prohibited as a single categorical value.

| Issue | Observed evidence | Magnitude | Required action |
| --- | --- | --- | --- |
| Skills data quality | required_skills raw string is unique per row; 119 rows contain repeated tokens. | 1,500 strings; 93 distinct tokens | Never one-hot raw strings; split tokens, de-duplicate within row, fit vocabulary on TRAIN. |

Không one-hot trực tiếp chuỗi raw: Nếu bạn biến mỗi chuỗi kỹ năng thành một cột one-hot, sẽ có hàng nghìn cột duy nhất (vì mỗi chuỗi là khác nhau), dẫn đến dữ liệu nhiễu và không dùng được.
Tách token (split tokens): Cần tách chuỗi kỹ năng thành các từ/tokens riêng lẻ (ví dụ: "Python, SQL, Machine Learning" → ["Python", "SQL", "Machine Learning"]).
Loại trùng trong từng dòng (de-duplicate within row): Nếu một kỹ năng lặp lại nhiều lần trong cùng một dòng, chỉ giữ một lần.
Xây vocabulary trên tập TRAIN (fit vocabulary on TRAIN): Tạo danh sách kỹ năng duy nhất từ tập train, sau đó encode dữ liệu theo danh sách này. Điều này đảm bảo mô hình chỉ học từ những kỹ năng đã thấy trong train, tránh mismatch khi gặp dữ liệu mới.
- **Important caveat**: pd.get_dummies is fine for a one-off analysis dataframe, but is not safe to reuse for the deployed model — the exact set of dummy columns it produces depends on which categories exist in the full dataset. A single new row from a form won't contain every category, so get_dummies would silently produce a mismatched column set. The actual modelling pipeline (Section 15) uses a proper ColumnTransformer instead, specifically to avoid this problem.
**7.4.**** ****Scaling **(Thay đổi quy mô):
- StandardScaler for Ridge/ElasticNet/SVR; not required for tree ensembles. Use model-specific pipelines rather than one universal preprocessing compromise.
**7.5.**** ****Feature Selection:**
- The project employs a strategic, two-phase approach to feature selection:
  - **Phase 1:** Begin with Lasso (L1 Regularization) to quickly eliminate "garbage" variables due to its simplicity and computational efficiency.
  - **Phase 2:** Subsequently, utilize Recursive Feature Elimination (RFE) paired with robust models like Random Forest and SVM to achieve higher accuracy and gain deeper insights into the interactions between specific skills.

This combined strategy is designed to combat the "Curse of Dimensionality"—a phenomenon occurring when the number of features is excessively high relative to the sample size (1,500 rows in this dataset), causing data sparsity and making the model prone to learning random noise instead of actual patterns.

- **Lasso (L1 Regularization - Embedded Method)**
  - **Mechanism:** Lasso incorporates a penalty term proportional to the sum of the absolute values of the coefficients into the loss function. This mathematical property shrinks the coefficients of non-contributing or redundant variables to exactly zero, effectively performing automated feature selection during model training.
  - **Application Strategy:** Use Lasso as an initial automated filtering step to prune "weak" features (those with an R2<0.01) among the dozens of AI skill columns.
  - **Primary Benefit:** Significantly increases Interpretability by highlighting the most influential predictors.

- **RFE (Recursive Feature Elimination - Wrapper Method)**
  - **Mechanism:** RFE works by training a base estimator (e.g., Random Forest), ranking features by their calculated importance, and iteratively removing the weakest features until the optimal subset remains.
  - **Primary Benefit:** Provides Resource Optimization by reducing dimensionality, leading to faster training times, lower memory consumption, and simplified maintenance.
  - **Technical Advantage:** Unlike Lasso, RFE is more effective at capturing feature interactions—such as the synergistic relationship between Job Title and Required Skills.
  - **Project Strategy:** For this AI market dataset, RFE will be integrated with TimeSeriesSplit (Expanding Window) to ensure that the selected features remain stable and predictive over the project's temporal range (Jan 2025 – Mar 2026).
**7.6. Feature Construction:**
- Feature Construction (Xây dựng đặc trưng mới):
Tạo biến "Salary-to-Experience Ratio" để hiểu giá trị của một năm kinh nghiệm trong các mảng AI khác nhau.
Skill Grouping: Nhóm các kỹ năng AI *(ví dụ: gom "PyTorch", "TensorFlow", "Keras" thành "Deep Learning Frameworks")* để giảm số lượng biến và tăng tính tổng quát.
- Cross-Validation: Luôn sử dụng K-Fold Cross-Validation trên tập Train để đảm bảo mô hình ổn định trước khi đánh giá cuối cùng trên tập Test.
Tập Test cuối cùng (Hold-out): Mar 2026 (chiếm 20% dữ liệu, dùng để đánh giá mô hình sau cùng sau khi đã chọn được hyperparameter tốt nhất).
Các Fold để Cross-Validation trên tập Train (Jan 2025 – Jan 2026):
Fold 1: Train (Jan 2025 – May 2025); Validation (Jun 2025).
Fold 2: Train (Jan 2025 – Jun 2025); Validation (Jul 2025).
Fold 3: Train (Jan 2025 – Jul 2025); Validation (Aug 2025).
Fold 4: Train (Jan 2025 – Aug 2025); Validation (Sep 2025).
Fold 5: Train (Jan 2025 – Sep 2025); Validation (Oct 2025).
Tiếp tục cho đến Feb 2026.
- Hướng dẫn thực hiện kỹ thuật (Logic Code): Để triển khai 5 folds này một cách chuẩn xác, bạn cần tuân thủ các quy tắc sau từ nguồn tài liệu:
Tham số n_splits: Thiết lập n_splits=5 trong hàm TimeSeriesSplit của thư viện Scikit-Learn.
Không xáo trộn (No Shuffling): Luôn đảm bảo shuffle=False để giữ đúng trình tự thời gian.
Tính toán thống kê: Mọi bước như chuẩn hóa (Scaling) hay điền khuyết (Imputation) phải được tính toán thông số (.fit) trên phần Train của từng fold và chỉ áp dụng (.transform) lên phần Validation tương ứng để tránh rò rỉ thông tin.
- Cách đánh giá kết quả từ 5 Folds: Sau khi chạy xong 5 folds, bạn cần thực hiện các bên dưới.
Tính điểm trung bình (Mean Score): Lấy trung bình cộng kết quả của 5 fold để có cái nhìn tổng quát về hiệu suất mô hình.
Kiểm tra độ lệch chuẩn (Std Dev):
Nếu độ lệch giữa các fold thấp: Mô hình ổn định.
Nếu độ lệch cao (High Variance): Mô hình đang nhạy cảm với dữ liệu và có nguy cơ quá khớp (overfitting).
So sánh với Baseline: Kết quả trung bình này phải cao hơn mô hình Dummy (mô hình dự đoán ngẫu nhiên) để đảm bảo mô hình có giá trị sử dụng.
- Residual Analysis (Phân tích phần dư): Sau khi huấn luyện, hãy kiểm tra biểu đồ phần dư để xem mô hình có dự đoán sai lệch nhiều ở các mức lương quá cao hay quá thấp hay không.
**7.7. Preprocessing Contract:**
- Configuration Contract
project:
   name: ai_job_market_salary_prediction
   seed: 42

 data:
   path: data/raw/ai_jobs_market_2025_2026.csv
   target: annual_salary_usd
   canonical_rename:
 	"AI Engineering": job_category

 split:
   strategy: temporal
   locked_test:
 	year: 2026
 	month: 3

 features:
   blocked: [job_id, salary_min_usd, salary_max_usd, salary_tier,
         	is_senior, is_remote_friendly, is_llm_role]
   contradictory: [years_of_experience, experience_level]
   phase2: [required_skills]

 metrics:
   primary: MAE
   secondary: [RMSE, R2, MedAE]

 artifacts:
   output_root: outputs
   bundle_path: artifacts/model_bundle.joblib
Configuration rule**: **Thresholds, allowed features, split dates, model search spaces and narrative acceptance criteria must live in versioned config files, not scattered as hard-coded constants across notebooks and Streamlit.

- Stage Failure Policy

| Failure | Pipeline response |
| --- | --- |
| Required column missing / wrong dtype | Fail closed; no training or serving. Emit schema error with column-level detail. |
| Invalid categorical token | Quarantine/drop if confidently corrupted and very low-volume; otherwise stop for rule decision. |
| Blocked feature enters X | Raise runtime error; invalidate experiment. |
| Preprocessing fitted before split | Invalidate run and rebuild from raw data. |
| Locked test used during tuning | Invalidate final performance estimate; create a new governed holdout if possible. |
| Unknown category at inference | Encode safely; set unseen-category flag; monitor rate. |
| Out-of-range numeric input | Predict only if policy allows; attach out-of-distribution / review flag. |
| Bundle reload changes prediction | Deployment release fails; do not serve. |

## **STAGE 8. Train-Test Split**
(PIC = Minh)
- Data followed time series:
The split was performed on the raw cleaned df (not the get_dummies df_corr), since the modelling pipeline does its own encoding internally via ColumnTransformer
training / testing: select all records of [posting_year = 2026, posting_month = 3] for testing (about 20%), the historical data for train (around 80%).
Train shape: approximately 1,199 rows × 11 columns
Test shape: approximately 300 rows × 11 columns
**Locked-test rule.  **2026-03 is not part of CV. Test labels are not inspected until the feature set, model class, hyperparameters, interval policy and narrative template are frozen.
- Each candidate model was wrapped in a single scikit-learn Pipeline containing a ColumnTransformer preprocessor, so the saved artifact is the entire pipeline — encoder and model together.

**BEFORE VS AFTER DATA PROCESSING**

| Aspect | Before Processing | After Processing |
| --- | --- | --- |
| Data Structure | Raw, 25 columns incl. IDs, leakage & free text | Clean, 11 modelling features, leakage-free |
| Corrupted Values | Header value leaked into a data row (hidden) | Identified & removed |
| Experience Signals | Two contradictory experience columns | Investigated; redundant column dropped |
| Encoding | Unsafe pd.get_dummies (analysis-only) | Deployable ColumnTransformer pipeline |
| Target Leakage | High risk (salary_min/max/tier present) | Fully eliminated |
| Model Readiness | Not Ready | Fully Ready — single deployable artifact |

This comparison highlights how a data-quality-first approach — going beyond nulls and duplicates — was essential to turning this dataset into a trustworthy modelling input.

## **STAGE 9. Model Training & Comparison**
(basic + ensemble model)

## .3 Baseline and Candidate Model Ladder

| Model | Role | Technical rationale |
| --- | --- | --- |
| DummyRegressor (median) | Non-ML floor | Any promoted model must materially outperform this. |
| Linear Regression | Transparent linear baseline | Detect whether relationships are mostly additive. |
| Ridge / ElasticNet | Regularized linear baseline | Stable with many one-hot features; coefficients are auditable. |
| Random Forest | Nonlinear bagging baseline | Robust tabular comparator; inspect overfit and inference size. |
| Extra Trees | High-variance-reduction tree ensemble | Strong tabular diagnostic; validate stability. |
| Gradient Boosting / HistGradientBoosting | Primary advanced scikit-learn candidate | Strong nonlinear tabular candidate; tune depth/learning rate/regularization. |
| SVR RBF | Optional diagnostic | Requires scaling; sparse high-dimensional OHE may be unfavorable. |
| CatBoost / LightGBM / XGBoost | Optional extension | Use only if dependency policy allows; compare fairly using same splits and metrics. |

## 6.4 Hyperparameter Tuning Rules
Tuning is nested inside the development data and temporal folds; the final test is never passed to a search object.
Prefer RandomizedSearchCV or a small bounded search space because the dataset has only ~1.2k development rows after holdout.
Optimize primary metric = negative MAE or a weighted multi-metric decision, while recording RMSE, R2 and fold variance.
Every trial logs model parameters, feature-set ID, split/fold IDs, seed, train time, inference time and metrics.
If two candidates are statistically/practically similar, select the simpler, more stable and more interpretable candidate.
## 6.5 Evaluation Metrics

| Metric | Formula | Priority | Interpretation |
| --- | --- | --- | --- |
| MAE | (1/n) * sum |y_i - yhat_i| | Primary | Directly interpretable as average absolute USD error; robust relative to RMSE. |
| RMSE | sqrt((1/n) * sum (y_i - yhat_i)^2) | Primary secondary | Penalizes large salary misses more strongly. |
| R2 | 1 - SS_res / SS_tot | Diagnostic | Fraction of variance explained on the evaluated split; can be misleading when data generation is synthetic. |
| Median AE | median(|y_i - yhat_i|) | Robust diagnostic | Shows typical error less affected by a few large residuals. |
| Interval coverage | mean(y_i in [L_i, U_i]) | If intervals enabled | Empirical coverage of prediction interval; not the same as parameter confidence interval. |
| Interval width | mean(U_i - L_i) | If intervals enabled | Coverage must be considered together with useful width. |

## 6.6 Model Promotion Criteria

| Gate | GO condition |
| --- | --- |
| Predictive value | MAE/RMSE materially beat Dummy and are stable across temporal folds. |
| Temporal generalization | Locked-test degradation is understood and acceptable; no hidden tuning against test. |
| Feature plausibility | No blocked fields; suspicious feature-family gains are documented by ablation. |
| Error slices | No severe unexplained degradation in adequately supported role/domain/location slices. |
| Complexity | Inference time, artifact size and maintenance burden justified by incremental performance. |
| Explainability | Global and local explanation available; no causal claims. |
| Reproducibility | Run can be reproduced from fingerprint + config + environment + seed. |
| Deployment equivalence | Reloaded bundle yields numerically equivalent predictions to offline pipeline. |

**9.1.** Five regression models were trained and compared on the identical train/test split:
Linear Regression — simplest interpretable baseline
Ridge Regression — linear plus regularization, handles many one-hot columns better than plain linear
Random Forest — nonlinear, handles mixed feature types without scaling
Gradient Boosting — usually the strongest of these on tabular data
SVM (RBF kernel) — nonlinear but distance-based, needs scaled inputs
**9.2.** Ensemble Model:
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV), especially for Gradient Boosting
Ensemble stacking of the strongest models
Re-investigating years_of_experience once a corrected, verified dataset is available
**9.3.** Need to read result from output, example:

| Model | MAE | RMSE | R² Score |
| --- | --- | --- | --- |
| Gradient Boosting | 14,522.78 | 25,808.91 | 0.852 |
| Random Forest | 14,742.45 | 27,639.93 | 0.830 |
| Ridge Regression | 25,347.58 | 35,734.20 | 0.716 |
| Linear Regression | 25,393.36 | 35,754.04 | 0.716 |
| SVM (RBF kernel) | 47,989.42 | 64,351.12 | 0.080 |

**Ex: Gradient Boosting achieved the best performance** (R² = 0.852), narrowly ahead of Random Forest (R² = 0.830). Both linear models performed identically and far behind the tree-based models, while the SVM landed close to a dummy-baseline result. This is not a tuning mistake — it reflects a structural mismatch: SVR with an RBF kernel measures distance in feature space, and after one-hot encoding this dataset has roughly 100 dimensions, most of them sparse binary indicators. Distance-based methods degrade badly in exactly that setting, while tree-based splits are unaffected by how many sparse dummy columns exist.

## **STAGE 10. Best Model Selection & Feature Importance Review**
**10.1. Best model: **Gradient Boosting (test R² = 0.852)
**10.2. Top Feature Importances:**
AI Engineering_AI Engineering (core domain): 0.7237 — dominant driver of the model's predictions
years_of_experience: 0.2056
AI Engineering_Security: 0.0119
AI Engineering_Robotics: 0.0092
AI Engineering_Architecture: 0.0076
All remaining features individually contribute well under 1% each
**Caution: **these two features alone (AI Engineering domain + years_of_experience) account for over 92% of the model's total feature importance. Since years_of_experience carries the contradictory, likely-synthetic signal flagged in Section 10, the model's strong R² should be reported honestly as a good fit to this specific dataset — not oversold as a fully causal, real-world explanation of AI salaries. Feature importance was deliberately inspected here rather than accepting the R² score at face value.

## **STAGE 11. Save Deployable Pipeline + Metadata**
- The winning pipeline was split into its preprocessing step and its final estimator, then saved as separate reusable artifacts:
preprocessor.pkl — the fitted preprocessing pipeline (encodes raw columns into model-ready features)
model.pkl — the trained Gradient Boosting estimator
feature_columns.json — the exact raw column order the preprocessor expects as input
metadata.json — best_model_name, test_r2, test_mae, category_options (valid values seen per categorical column during training), numeric_ranges (min/max seen for numeric columns), and ordinal_order (the education ranking)
- To save the full pipeline and supporting metadata as deployable artifacts: Saving category_options and numeric_ranges alongside the model lets the Streamlit UI build dropdowns and sliders that only offer values the model actually learned from, instead of guessing or allowing invalid inputs.

## **STAGE 12. Streamlit Salary Prediction Dashboard**
- Visualize data and inllustrate pie, bar, line, statistics graph/ chart and table followed above 12 stages of pineline on Streamlit:
Performing data quality checks that go beyond nulls and duplicates — catching corrupted values hidden inside categorical columns
Investigating and reasoning critically about contradictory features before trusting either of them
Applying the correct ordinal vs nominal encoding strategy for different categorical variables
Building deployment-safe preprocessing with ColumnTransformer + Pipeline instead of ad-hoc pd.get_dummies
Fairly comparing multiple regression algorithms on an identical train/test split
Interpreting feature importance critically rather than accepting a strong R² at face value
Packaging a full ML pipeline and supporting metadata for real-time deployment via Streamlit
- Advanced Analytics
Predicting a salary range/confidence interval instead of a single point estimate: build an interactive Streamlit dashboard for real-time salary prediction
Geographic salary-adjustment modelling (cost-of-living aware)
Skill-based salary premium analysis using required_skills via NLP

# D – CODE IMPLEMENTATION
**Dataset: **ai_jobs_market_2025_2026.csv (1,500 AI job postings, 25 attributes)
## **D.1. salary_predict_corrected.ipynb File**
***# Import the libraries and Load the dataset***
import pandas as pd
import numpy as np

# NOTE: update this path to match your local machine
df = pd.read_csv("C:\\Users\\mayank manjhi\\Downloads\\list_projects\\Ai_job_market\\data\\ai_jobs_market_2025_2026.csv")

***# Data's Initial Inspection  Load the raw dataset and review a sample of records to understand its structure.***
df.head(5)

df.info()

df.describe().round(2)

***# Data Quality Checks  Check nulls, duplicates, **and** sanity-check every categorical column's values *before* doing any correlation or feature work. This step was missing before, and it's what catches corrupted values that `.isnull()` and `.duplicated()` won't.***
print("Null values in each column:\n", df.isnull().sum())
print("=" * 50)
print("Duplicate rows:", df.duplicated().sum())

# Sanity-check every categorical column's values, not just null/duplicate counts.
# This is what catches corrupted or unexpected values sitting inside otherwise "clean" columns.
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
print(f"\n--- {col} ({df[col].nunique()} unique values) ---")
print(df[col].value_counts())

*****Finding:** the `AI Engineering` column contains a value `"job_category"` (count = 1) — that is literally the column header leaked into the data as a row value (job_id `AIJOB1205`). This would have silently corrupted any encoding/correlation work downstream. Fixing it now, before feature selection.***
bad_rows = df[df["AI Engineering"] == "job_category"]
print(f"Corrupted rows found: {len(bad_rows)}")
display(bad_rows)

# Only 1 row out of 1500 — drop it rather than guess/impute a category.
df = df[df["AI Engineering"] != "job_category"].reset_index(drop=True)
print("Row count after cleanup:", len(df))

***# Investigate Suspicious Relationships Before Modeling  `years_of_experience` and `experience_level` should tell a consistent story (more experience -> higher pay). Check that assumption before trusting either column as a feature.***
print("Mean salary by experience_level (categorical bucket):")
print(df.groupby("experience_level")["annual_salary_usd"].mean().round(0))
print()
print("Mean salary by years_of_experience (raw numeric):")
print(df.groupby("years_of_experience")["annual_salary_usd"].mean().round(0))

*****Finding:** `experience_level` shows salaries flat across Entry/Mid/Senior/Lead (~$193k-196k each), while `years_of_experience` shows a strong *downward* trend (1 yr -> ~$326k, 15 yrs -> ~$90k). These two columns directly contradict each other and cannot both be reliable signals of real-world pay-for-experience.  This is a data quality issue in the source dataset (very likely a synthetic dataset with a generation bug), not something to encode around silently. Decision taken here: keep `years_of_experience` as a feature since it's the more granular of the two, drop the redundant `experience_level` bucket, and flag this relationship as unreliable/synthetic rather than trusting it as a real causal signal in any write-up.***
***# Feature Selection  Remove columns that either leak the target (derived directly from salary) or aren't useful as modeling inputs (IDs, free-text, redundant buckets).***
leak_and_unused_cols = [
"job_id",                 # identifier, not a feature
"salary_min_usd",         # leaks target (derived from same salary structure)
"salary_max_usd",         # leaks target
"salary_tier",            # leaks target (binned version of the target)
"experience_level",       # redundant with years_of_experience, and contradicts it (see above)
"required_skills",        # free text, would need separate NLP treatment
"posting_year",
"posting_month",
"is_senior",
"is_remote_friendly",
"is_llm_role",
"ai_salary_premium_pct",
"demand_growth_yoy_pct",
]

df = df.drop(columns=leak_and_unused_cols)
df.head()

***# Encoding for Correlation Analysis  Build a *separate* encoded copy for correlation analysis (`df_corr`). This uses `pd.get_dummies`, which is fine for a one-off analysis dataframe but is **not safe to reuse for the deployed model** — the exact set of dummy columns it produces depends on which categories exist in the full dataset. A single new row (e.g. from a form) won't contain all those categories, so `get_dummies` would silently produce a mismatched column set. The modeling section further down uses a proper `ColumnTransformer` instead, for exactly this reason.  Corrected encoding logic (same reasoning as before): - **Ordinal columns** (have a real order) -> map to an explicit rank, not LabelEncoder's alphabetical default. `education_required` is the one ordinal column here. - **Nominal columns** (no real order) -> one-hot encode, regardless of how many unique values they have.***
df_corr = df.copy()
