*Part 3 of 5*
# Ordinal: education has a real ranking, so encode it as one
education_order = {
"Bootcamp/Self-taught": 0,
"Associate's": 1,
"Bachelor's": 2,
"Master's": 3,
"PhD": 4,
}
df_corr["education_required"] = df_corr["education_required"].map(education_order)

# Nominal: one-hot encode everything else that's still categorical, regardless of cardinality
nominal_cols_for_corr = df_corr.select_dtypes(include=["object"]).columns.tolist()
print("One-hot encoding (correlation view only):", nominal_cols_for_corr)

df_corr = pd.get_dummies(df_corr, columns=nominal_cols_for_corr, drop_first=False)
df_corr.head()

***# Correlation Analysis (single pass, on the corrected encoding)***
target_corr = df_corr.corr(numeric_only=True)["annual_salary_usd"].sort_values(ascending=False)
target_corr

import matplotlib.pyplot as plt

# A full heatmap is unreadable once job_title/city/etc are one-hot encoded (90+ columns).
# A sorted bar chart of correlation with the target is the readable equivalent.
top_corr = target_corr.drop("annual_salary_usd").reindex(
target_corr.drop("annual_salary_usd").abs().sort_values(ascending=False).index
).head(25)

plt.figure(figsize=(10, 8), dpi=100)
colors = ["#d62728" if v < 0 else "#2ca02c" for v in top_corr.values]
plt.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 25 Features by |Correlation| with annual_salary_usd")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

*****Note on interpreting this correlation table:** one-hot dummy columns (e.g. `AI Engineering_Security`, `job_title_LLM Engineer`) are binary indicators, so their correlation with salary is meaningful — it tells you whether *belonging to that specific category* is associated with higher or lower pay. That is different from the old approach, where a single LabelEncoder column mixed all categories into one arbitrary numeric scale and produced a correlation number that depended on alphabetical order, not on the actual data. Any `job_title`/`AI Engineering` correlation you read from this table now reflects a real category-level relationship, not an encoding artifact.***
***# Train/Test Split for Modeling  Splitting the **raw cleaned `df`** (not `df_corr`) — the modeling pipeline below does its own encoding internally via `ColumnTransformer`, so it needs the original untouched columns, not the `get_dummies` output.***
from sklearn.model_selection import train_test_split

X = df.drop(columns=["annual_salary_usd"])
y = df["annual_salary_usd"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

***# Model Selection  Compare a few baseline regressors on the same train/test split before committing to one. Including a `DummyRegressor` (predicts the mean every time) as a sanity-check floor — any real model should clearly beat it, otherwise the features aren't adding value.  Each model is wrapped in a `Pipeline` with a `ColumnTransformer` preprocessor (ordinal-encode `education_required`, one-hot encode the rest with `handle_unknown="ignore"`, pass numeric columns through unchanged). This means the **saved artifact is the whole pipeline** — encoder and model together — so it can take a single raw new row later (e.g. from a Streamlit form) without needing a separate encoder kept in sync by hand.  Models compared: - **Linear Regression** — simplest interpretable baseline - **Ridge Regression** — linear + regularization, handles the many one-hot columns better than plain linear - **Random Forest** — nonlinear, handles mixed feature types without scaling - **Gradient Boosting** — usually the strongest of these on tabular data, slower to train - **SVM (RBF kernel)** — nonlinear, but distance-based so it needs scaled inputs; often struggles on high-dimensional one-hot data like this without careful tuning***
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ordinal_col = ["education_required"]
ordinal_categories = [["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"]]
nominal_cols = [c for c in X.select_dtypes(include=["object"]).columns if c not in ordinal_col]
numeric_cols = [c for c in X.columns if c not in nominal_cols + ordinal_col]

print("Ordinal:", ordinal_col)
print("Nominal (one-hot):", nominal_cols)
print("Numeric (passthrough):", numeric_cols)

def make_preprocessor():
# Fresh ColumnTransformer per model -- avoids sharing fitted state across pipelines.
return ColumnTransformer(
transformers=[
("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_col),
("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),
],
remainder="passthrough",
)

models = {

"Linear Regression": Pipeline([
("preprocess", make_preprocessor()),
("model", LinearRegression()),
]),
"Ridge Regression": Pipeline([
("preprocess", make_preprocessor()),
("model", Ridge(alpha=1.0, random_state=42)),
]),
"Random Forest": Pipeline([
("preprocess", make_preprocessor()),
("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
]),
"Gradient Boosting": Pipeline([
("preprocess", make_preprocessor()),
("model", GradientBoostingRegressor(random_state=42)),
]),
"SVM (RBF kernel)": Pipeline([
("preprocess", make_preprocessor()),
("scale", StandardScaler()),
("model", SVR(kernel="rbf", C=1000, epsilon=1000)),
]),
}

results = []
fitted_models = {}

for name, model in models.items():
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
results_df.round(3)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5), dpi=100)
plt.barh(results_df["model"][::-1], results_df["R2"][::-1], color="#1f77b4")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("R2 on test set")
plt.title("Model Comparison — Test Set R2")
plt.tight_layout()
plt.show()

*****Reading the comparison:** the best model here is whichever has the highest R2 / lowest RMSE — check the table above rather than assuming Gradient Boosting wins by default, since that depends on this specific split and feature set.  **On the SVM result specifically:** don't be surprised if it lands at or below the dummy baseline. `C`/`epsilon` tuning outside this notebook only reached R2 ~0.08, nowhere near the tree models' ~0.85. This isn't a tuning mistake, it's a structural mismatch: SVR with an RBF kernel measures distance in feature space, and this dataset is ~100 dimensions after one-hot encoding, most of them sparse binary indicators. Distance-based methods degrade in exactly that setting, while tree-based splits don't care how many sparse dummy columns exist.  **Caveat carried over from the correlation step:** `years_of_experience` has a strong negative relationship with salary that contradicts `experience_level`'s (dropped) flat relationship. If the best model is leaning heavily on `years_of_experience`, that's the model learning a pattern that may be a synthetic-data artifact rather than a real-world signal. Check feature importance below before trusting this model on new data — don't just take the R2 at face value.***
best_model_name = results_df.iloc[0]["model"]
best_pipeline = fitted_models[best_model_name]
print("Best model by R2:", best_model_name)

preprocess_step = best_pipeline.named_steps["preprocess"]
feature_names = preprocess_step.get_feature_names_out()
final_estimator = best_pipeline.named_steps["model"]

if hasattr(final_estimator, "feature_importances_"):
importances = pd.Series(final_estimator.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False).head(15)
print("\nTop 15 feature importances:")
print(importances.round(4))
elif hasattr(final_estimator, "coef_"):
coefs = pd.Series(final_estimator.coef_, index=feature_names)
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)
print("\nTop 15 coefficients by magnitude:")
print(coefs.round(2))
else:
print("No importances/coefficients exposed by this model type.")

***# Save the Best Model as a Deployable Artifact  The object saved below is the **entire pipeline** — preprocessing *and* model together — not just the raw estimator. That's the point of building it with `ColumnTransformer` earlier: hand it a single raw row with the original column names (`job_title`, `AI Engineering`, `years_of_experience`, `education_required`, `city`, `country`, `remote_work`, `company_size`, `industry`, `demand_score`, `benefits_score_10`) and it encodes + predicts in one call. No separate encoder to keep in sync by hand — which matters once this gets used from something like a Streamlit app.  Also saving a small `model_metadata.json` alongside it: the valid category values and numeric ranges seen during training, so a UI can build dropdowns/sliders that only offer values the model actually knows about, instead of guessing.***
import joblib
import json
from pathlib import Path
from sklearn.pipeline import Pipeline as SkPipeline

# Where to save artifacts — adjust if you want a different location
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

# Split the winning pipeline into its preprocessing steps and its final estimator.
# Works regardless of which model won (e.g. SVM has an extra "scale" step, trees don't).
steps = dict(best_pipeline.named_steps)
model = steps.pop("model")
preprocessor = SkPipeline(list(steps.items()))

# 1. The fitted preprocessor (encodes raw columns -> model-ready features)
joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")

# 2. The trained model (fits on the preprocessor's output)
joblib.dump(model, artifacts_dir / "model.pkl")

# 3. The exact raw column order the preprocessor expects as input
feature_columns = X_train.columns.tolist()
with open(artifacts_dir / "feature_columns.json", "w") as f:
json.dump(feature_columns, f, indent=2)

# 4. Everything a UI needs to build valid inputs + show model context
metadata = {
"best_model_name": best_model_name,
"test_r2": float(results_df.iloc[0]["R2"]),
"test_mae": float(results_df.iloc[0]["MAE"]),
"category_options": {
col: sorted(X_train[col].dropna().unique().tolist())
for col in nominal_cols + ordinal_col
},
"numeric_ranges": {
col: {"min": int(X_train[col].min()), "max": int(X_train[col].max())}
for col in numeric_cols
},
"ordinal_order": {"education_required": ordinal_categories[0]},
}
with open(artifacts_dir / "metadata.json", "w") as f:
json.dump(metadata, f, indent=2)

print("Saved:", sorted(p.name for p in artifacts_dir.iterdir()))
print(f"Best model: {best_model_name}  (test R2 = {metadata['test_r2']:.3f})")
