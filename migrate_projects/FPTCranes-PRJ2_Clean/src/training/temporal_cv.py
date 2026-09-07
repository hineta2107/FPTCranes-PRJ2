from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def monthly_temporal_folds(frame: pd.DataFrame, n_folds: int = 5) -> list[dict[str, Any]]:
    month_key = frame["posting_year"].astype(int) * 100 + frame["posting_month"].astype(int)
    months = sorted(month_key.unique().tolist())
    if len(months) < n_folds + 2:
        raise RuntimeError("Not enough distinct development months for expanding-window temporal CV.")
    validation_months = months[-n_folds:]
    folds: list[dict[str, Any]] = []
    for number, validation_month in enumerate(validation_months, 1):
        train_idx = frame.index[month_key < validation_month].to_numpy()
        validation_idx = frame.index[month_key == validation_month].to_numpy()
        if len(train_idx) == 0 or len(validation_idx) == 0:
            raise RuntimeError(f"Invalid temporal fold for month {validation_month}.")
        folds.append(
            {
                "fold": number,
                "validation_month": int(validation_month),
                "train_idx": train_idx,
                "validation_idx": validation_idx,
            }
        )
    return folds


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }
