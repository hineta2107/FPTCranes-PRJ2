"""Tests validating Stage 8-10 requirements against existing code behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import BLOCKED_FEATURES, MODEL_FEATURES, LOCKED_YEAR, LOCKED_MONTH
from src.models import TrainingSelection
from src.training import (
    build_preprocessor,
    model_catalog,
    monthly_temporal_folds,
    regression_metrics,
    select_best_model,
    out_of_fold_absolute_errors,
)


# ── T004: Temporal split boundary ────────────────────────────────────────────

def test_temporal_split_boundary():
    """TRAIN contains only records before March 2026, TEST only March 2026."""
    raw = pd.read_csv("data/ai_jobs_market_2025_2026.csv", low_memory=False)
    # Stage 4 cleaning: remove corrupted row where job_category == "job_category"
    mask = raw["job_title"] != "job_title"
    clean = raw[mask].copy()
    clean["posting_year"] = clean["posting_year"].astype(int)
    clean["posting_month"] = clean["posting_month"].astype(int)

    dev = clean[
        ~((clean["posting_year"] == LOCKED_YEAR) & (clean["posting_month"] == LOCKED_MONTH))
    ]
    locked = clean[
        (clean["posting_year"] == LOCKED_YEAR) & (clean["posting_month"] == LOCKED_MONTH)
    ]

    # Actual counts depend on corrupted row location; use ranges from spec
    assert 1195 <= len(dev) <= 1205, f"TRAIN out of range: {len(dev)}"
    assert 295 <= len(locked) <= 305, f"TEST out of range: {len(locked)}"
    assert locked["posting_year"].unique().tolist() == [LOCKED_YEAR]
    assert locked["posting_month"].unique().tolist() == [LOCKED_MONTH]
    # No dev rows should be in the locked period
    dev_in_locked = dev[
        (dev["posting_year"] == LOCKED_YEAR) & (dev["posting_month"] == LOCKED_MONTH)
    ]
    assert len(dev_in_locked) == 0, "Dev data leaked into locked period"


# ── T005: Blocked features absent from MODEL_FEATURES ───────────────────────

def test_blocked_features_absent_from_model_features():
    """All 12 BLOCKED_FEATURES must not appear in MODEL_FEATURES."""
    overlap = set(BLOCKED_FEATURES) & set(MODEL_FEATURES)
    assert len(overlap) == 0, f"Blocked features found in MODEL_FEATURES: {overlap}"
    assert len(BLOCKED_FEATURES) == 12


# ── T006: Preprocessor includes CountVectorizer for required_skills ──────────

def test_preprocessor_includes_skills_vectorizer():
    """build_preprocessor() must include a CountVectorizer for required_skills."""
    preprocessor = build_preprocessor(scale_numeric=True)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "skills" in transformer_names, (
        f"Missing 'skills' transformer. Found: {transformer_names}"
    )
    skills_transformer = dict(
        (name, t) for name, t, _ in preprocessor.transformers
    )["skills"]
    assert "CountVectorizer" in type(skills_transformer).__name__


# ── T007: Preprocessor has 4 transformer steps ──────────────────────────────

def test_preprocessor_has_four_transformers():
    """ColumnTransformer must have nominal, education, numeric, skills."""
    preprocessor = build_preprocessor(scale_numeric=True)
    names = [name for name, _, _ in preprocessor.transformers]
    assert len(names) == 4, f"Expected 4 transformers, got {len(names)}: {names}"
    for expected in ("nominal", "education", "numeric", "skills"):
        assert expected in names, f"Missing transformer: {expected}"


# ── T008: Model catalog returns 5 models ─────────────────────────────────────

def test_model_catalog_has_five_models():
    """Catalog must contain Dummy, Linear, Ridge, RF, GB (SVR removed)."""
    catalog = model_catalog()
    assert len(catalog) == 5, f"Expected 5 models, got {len(catalog)}: {list(catalog)}"
    expected_names = {
        "Dummy (Median)",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    assert set(catalog.keys()) == expected_names


# ── T009: regression_metrics returns MAE, RMSE, R2, MedAE ───────────────────

def test_regression_metrics_returns_all_four():
    """regression_metrics() must return dict with MAE, RMSE, R2, MedAE."""
    y_true = pd.Series([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    metrics = regression_metrics(y_true, y_pred)
    for key in ("MAE", "RMSE", "R2", "MedAE"):
        assert key in metrics, f"Missing metric key: {key}"
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] >= metrics["MAE"]


# ── T010: monthly_temporal_folds produces expanding-window folds ─────────────

def test_temporal_folds_are_expanding():
    """Each fold's train set must be a strict subset of the next fold's."""
    raw = pd.read_csv("data/ai_jobs_market_2025_2026.csv", low_memory=False)
    mask = raw["job_title"] != "job_title"
    clean = raw[mask].copy()
    clean["posting_year"] = clean["posting_year"].astype(int)
    clean["posting_month"] = clean["posting_month"].astype(int)
    dev = clean[
        ~((clean["posting_year"] == LOCKED_YEAR) & (clean["posting_month"] == LOCKED_MONTH))
    ].sort_values(["posting_year", "posting_month"]).reset_index(drop=True)

    folds = monthly_temporal_folds(dev, n_folds=5)
    assert len(folds) == 5
    for i in range(1, len(folds)):
        prev_train = set(folds[i - 1]["train_idx"])
        curr_train = set(folds[i]["train_idx"])
        assert prev_train < curr_train, (
            f"Fold {i} train is not a strict superset of fold {i-1}"
        )


# ── Shared fixture: load pipeline-prepared dev data ──────────────────────────

@pytest.fixture(scope="module")
def dev_data():
    """Load the development partition already prepared by Stages 1-7."""
    dev = pd.read_csv(
        "outputs/02_data_ready_for_machine_learning/train_raw_model_input.csv",
        low_memory=False,
    )
    return dev.sort_values(["posting_year", "posting_month"]).reset_index(drop=True)


# ── T011: select_best_model returns TrainingSelection ────────────────────────

def test_select_best_model_returns_training_selection(dev_data):
    """select_best_model() must return a TrainingSelection dataclass."""
    from src.training import compare_models
    folds = monthly_temporal_folds(dev_data, n_folds=5)
    _, comparison = compare_models(dev_data, folds)
    selection = select_best_model(dev_data, folds, comparison)
    assert isinstance(selection, TrainingSelection)
    assert selection.model_name in model_catalog()
    assert selection.estimator is not None


# ── T012: out_of_fold_absolute_errors returns positive array ─────────────────

def test_out_of_fold_errors_positive(dev_data):
    """out_of_fold_absolute_errors() returns positive values with valid interval."""
    from src.training import compare_models
    folds = monthly_temporal_folds(dev_data, n_folds=5)
    _, comparison = compare_models(dev_data, folds)
    selection = select_best_model(dev_data, folds, comparison)
    errors = out_of_fold_absolute_errors(dev_data, folds, selection)
    assert len(errors) > 0
    assert all(e >= 0 for e in errors)
    interval_q90 = float(np.quantile(errors, 0.90))
    assert interval_q90 > 0
