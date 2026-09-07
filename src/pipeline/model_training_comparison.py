"""Stage 9: Model Training & Comparison."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models import PipelinePaths
from src.training import compare_models, monthly_temporal_folds
from src.utils.pipeline_plots import plot_model_comparison

LOGGER = logging.getLogger("ml_pipeline")


def model_training_comparison(
    dev: pd.DataFrame, paths: PipelinePaths
) -> tuple[pd.DataFrame, list[Any], pd.DataFrame]:
    LOGGER.info("Stage 9: starting model training & comparison on %d dev rows", len(dev))
    dev = dev.sort_values(["posting_year", "posting_month"]).reset_index(drop=True)
    LOGGER.debug("Dev data sorted by posting date, shape=%s", dev.shape)
    folds = monthly_temporal_folds(dev, n_folds=5)
    LOGGER.info("Temporal folds ready: %d expanding-window splits", len(folds))
    fold_metrics, comparison = compare_models(dev, folds)
    leader = comparison.iloc[0]
    LOGGER.info(
        "Model comparison complete: leader=%s CV_MAE=%.2f CV_R2=%.4f",
        leader["model"],
        leader["CV_MAE_mean"],
        leader["CV_R2_mean"],
    )
    fold_metrics.to_csv(
        paths.comparison / "09_model_comparison_fold_metrics.csv", index=False
    )
    comparison.to_csv(
        paths.comparison / "09_model_comparison_temporal_cv.csv", index=False
    )
    LOGGER.debug("Saved fold metrics and comparison CSVs to %s", paths.comparison)
    plot_model_comparison(comparison, paths.comparison)
    LOGGER.info("Stage 9 complete: %d models compared, charts saved", len(comparison))
    return dev, folds, comparison
