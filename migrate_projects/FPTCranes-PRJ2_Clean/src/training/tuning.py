from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.constants import SEED, TARGET
from src.training.build_preprocessor import build_preprocessor


def _score_candidates(train_df: pd.DataFrame, folds: list[dict[str, Any]], feature_columns: list[str], candidates: list[dict[str, Any]], family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, params in enumerate(candidates, 1):
        fold_mae: list[float] = []
        for fold in folds:
            tr = train_df.loc[fold["train_idx"]]
            va = train_df.loc[fold["validation_idx"]]
            if family == "Random Forest":
                estimator = RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
            else:
                estimator = GradientBoostingRegressor(random_state=SEED, **params)
            pipe = Pipeline(
                [
                    ("preprocessor", build_preprocessor(feature_columns, False)),
                    ("model", estimator),
                ]
            )
            pipe.fit(tr[feature_columns], tr[TARGET])
            pred = pipe.predict(va[feature_columns])
            fold_mae.append(float(mean_absolute_error(va[TARGET], pred)))
        rows.append(
            {
                "candidate_id": candidate_id,
                **params,
                "CV_MAE_mean": float(np.mean(fold_mae)),
                "CV_MAE_std": float(np.std(fold_mae)),
            }
        )
    return pd.DataFrame(rows).sort_values("CV_MAE_mean").reset_index(drop=True)


def tune_random_forest(train_df: pd.DataFrame, folds: list[dict[str, Any]], feature_columns: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    # 4 candidates theo yêu cầu báo cáo: n_estimators, min_samples_leaf, max_features, max_depth
    candidates = [
        {"n_estimators": 80,  "min_samples_leaf": 1, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 100, "min_samples_leaf": 2, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.7, "max_depth": 20},
        {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.7, "max_depth": 20},
    ]
    result = _score_candidates(train_df, folds, feature_columns, candidates, "Random Forest")
    best = result.iloc[0]
    params = {
        "n_estimators": int(best["n_estimators"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": float(best["max_features"]),
        "max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
    }
    return params, result

def tune_gradient_boosting(train_df: pd.DataFrame, folds: list[dict[str, Any]], feature_columns: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = [
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2, "loss": "huber"},
        {"n_estimators": 250, "learning_rate": 0.04, "max_depth": 2, "loss": "huber"},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "loss": "huber"},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2, "loss": "squared_error"},
    ]
    result = _score_candidates(train_df, folds, feature_columns, candidates, "Gradient Boosting")
    best = result.iloc[0]
    params = {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "loss": str(best["loss"]),
    }
    return params, result
