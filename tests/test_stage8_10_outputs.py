"""Phase 3 — Output verification tests for Stage 8-10 pipeline artifacts.

Each test validates an existing pipeline output file against spec requirements.
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest

from src.constants.pipeline import BLOCKED_FEATURES

# ── Paths ────────────────────────────────────────────────────────────────────

ML_READY = pathlib.Path("outputs/02_data_ready_for_machine_learning")
COMPARISON = pathlib.Path("outputs/03_model_comparison")
BEST = pathlib.Path("outputs/04_best_model_and_feature_importance")
ARTIFACTS = pathlib.Path("artifacts")


# ── T014: split summary CSV ─────────────────────────────────────────────────

def test_split_summary_csv_row_counts():
    p = ML_READY / "08_split_summary.csv"
    assert p.exists(), f"{p} missing"
    df = pd.read_csv(p)
    train_row = df.loc[df["partition"] == "TRAIN_DEV"]
    test_row = df.loc[df["partition"] == "LOCKED_TEST"]
    assert 1195 <= int(train_row["rows"].iloc[0]) <= 1210
    assert 295 <= int(test_row["rows"].iloc[0]) <= 305


# ── T015: encoded feature names ─────────────────────────────────────────────

def test_encoded_feature_names_count():
    p = ML_READY / "08_encoded_feature_names.csv"
    assert p.exists(), f"{p} missing"
    df = pd.read_csv(p)
    assert 150 <= len(df) <= 250, f"Unexpected feature count: {len(df)}"


# ── T016: blocked features absent from train input ──────────────────────────

def test_train_input_no_blocked_columns():
    """Blocked features (except posting_year/month used for split) absent from model input."""
    p = ML_READY / "train_raw_model_input.csv"
    assert p.exists()
    cols = set(pd.read_csv(p, nrows=0).columns)
    # posting_year and posting_month are kept for temporal split, not as model features
    split_cols = {"posting_year", "posting_month"}
    model_blocked = set(BLOCKED_FEATURES) - split_cols
    leaked = cols & model_blocked
    assert not leaked, f"Blocked features in train input: {leaked}"


# ── T017: model comparison has 6 rows ───────────────────────────────────────

def test_comparison_csv_has_six_models():
    p = COMPARISON / "09_model_comparison_temporal_cv.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 6, f"Expected 6 model rows, got {len(df)}"
    for col in ["CV_MAE_mean", "CV_RMSE_mean", "CV_R2_mean"]:
        assert col in df.columns, f"Missing column {col}"


# ── T018: non-Dummy beats Dummy baseline ────────────────────────────────────

def test_non_dummy_beats_dummy():
    df = pd.read_csv(COMPARISON / "09_model_comparison_temporal_cv.csv")
    dummy_mae = df.loc[df["model"].str.contains("Dummy", case=False), "CV_MAE_mean"].values[0]
    best_other = df.loc[~df["model"].str.contains("Dummy", case=False), "CV_MAE_mean"].min()
    assert best_other < dummy_mae, "No model beat Dummy baseline"


# ── T019: comparison chart exists ───────────────────────────────────────────

def test_comparison_chart_exists():
    assert (COMPARISON / "model_comparison_cv_r2.png").exists()


# ── T020: locked test metrics pass thresholds ───────────────────────────────

def test_locked_test_metrics_thresholds():
    p = BEST / "10_final_locked_test_metrics.csv"
    assert p.exists()
    df = pd.read_csv(p)
    r2 = df["R2"].iloc[0] if "R2" in df.columns else df.iloc[0]["R2"]
    mae = df["MAE"].iloc[0] if "MAE" in df.columns else df.iloc[0]["MAE"]
    assert r2 >= 0.75, f"R² {r2} < 0.75 threshold"
    assert mae <= 20_000, f"MAE {mae} > 20,000 threshold"


# ── T021: feature importance concentration ──────────────────────────────────

def test_top2_feature_importance_concentration():
    p = BEST / "10_encoded_feature_importance.csv"
    assert p.exists()
    df = pd.read_csv(p).sort_values("importance", ascending=False)
    top_job = df[df["encoded_feature"].str.startswith("nominal__job_category")]["importance"].sum()
    top_exp = df[df["encoded_feature"].str.contains("years_of_experience")]["importance"].sum()
    assert top_job + top_exp > 0.90, f"Concentration {top_job + top_exp:.2%} < 90%"


# ── T022: permutation importance columns ────────────────────────────────────

def test_permutation_importance_columns():
    p = BEST / "10_raw_feature_permutation_importance.csv"
    assert p.exists()
    df = pd.read_csv(p)
    for col in ["raw_feature", "importance_mean", "importance_std"]:
        assert col in df.columns, f"Missing column {col}"


# ── T023: caveats JSON has 5 required keys ──────────────────────────────────

def test_caveats_json_keys():
    p = BEST / "10_interpretation_caveats.json"
    assert p.exists()
    data = json.loads(p.read_text())
    required = {
        "two_feature_concentration",
        "synthetic_signal_warning",
        "no_causal_claims",
        "honest_r2_framing",
        "svm_structural_mismatch",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing caveats: {missing}"


# ── T024: model bundle loads and predicts ───────────────────────────────────

def test_model_bundle_predict():
    bundle = joblib.load(ARTIFACTS / "model_bundle.joblib")
    # Use first row of actual training data as sample
    train = pd.read_csv(ML_READY / "train_raw_model_input.csv", nrows=1)
    meta = json.loads((ARTIFACTS / "metadata.json").read_text())
    features = meta["model_features"]
    sample = train[features]
    pred = bundle.predict(sample)
    assert pred.shape[0] == 1
    assert np.isfinite(pred[0])


# ── T025: metadata.json required keys ───────────────────────────────────────

def test_metadata_required_keys():
    meta = json.loads((ARTIFACTS / "metadata.json").read_text())
    for key in ["model_features", "model_name", "education_order", "prediction_interval_abs_error_q90"]:
        assert key in meta, f"Missing metadata key: {key}"


# ── T026: unknown category handling ─────────────────────────────────────────

def test_unknown_category_no_crash():
    bundle = joblib.load(ARTIFACTS / "model_bundle.joblib")
    # Start from real row, then inject unseen category in string columns
    train = pd.read_csv(ML_READY / "train_raw_model_input.csv", nrows=1)
    meta = json.loads((ARTIFACTS / "metadata.json").read_text())
    features = meta["model_features"]
    sample = train[features].copy()
    # Replace string columns with unseen values
    for col in sample.select_dtypes(include="object").columns:
        sample[col] = "NEVER_SEEN_CATEGORY_XYZ"
    pred = bundle.predict(sample)
    assert np.isfinite(pred[0])


# ── T027: interval_q90 positive ─────────────────────────────────────────────

def test_interval_q90_positive():
    meta = json.loads((ARTIFACTS / "metadata.json").read_text())
    q90 = meta["prediction_interval_abs_error_q90"]
    assert q90 > 0, f"interval_q90={q90} not positive"
