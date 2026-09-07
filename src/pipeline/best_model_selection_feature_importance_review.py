"""Stage 10: Best Model Selection & Feature Importance Review."""

from __future__ import annotations

from typing import Any
import json
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.constants import MODEL_FEATURES, SEED, TARGET
from src.models import PipelinePaths, TrainingSelection
from src.training import (
    build_pipeline,
    out_of_fold_absolute_errors,
    regression_metrics,
    select_best_model,
)
from src.utils.pipeline_plots import plot_best_outputs

LOGGER = logging.getLogger("ml_pipeline")


def best_model_selection_feature_importance_review(
    dev: pd.DataFrame,
    locked: pd.DataFrame,
    folds: list[Any],
    comparison: pd.DataFrame,
    paths: PipelinePaths,
) -> tuple[
    str,
    Any,
    np.ndarray,
    dict[str, float],
    Any,
    TrainingSelection,
    float,
]:
    selection = select_best_model(dev, folds, comparison)
    selected_name = selection.model_name
    selection.tuning_results.to_csv(
        paths.best / "10_best_model_tuning_results.csv", index=False
    )

    best_pipeline = build_pipeline(selection.estimator, selection.scale_numeric)
    LOGGER.info("Fitting selected model on all development rows: %s", selected_name)
    best_pipeline.fit(dev[list(MODEL_FEATURES)], dev[TARGET])
    locked_predictions = best_pipeline.predict(locked[list(MODEL_FEATURES)])
    final_metrics = regression_metrics(locked[TARGET], locked_predictions)
    LOGGER.info(
        "Locked-test evaluation complete: MAE=%.2f RMSE=%.2f R2=%.4f",
        final_metrics["MAE"],
        final_metrics["RMSE"],
        final_metrics["R2"],
    )
    pd.DataFrame(
        [
            {
                "selected_model": selected_name,
                **final_metrics,
                "selection_basis": "Lowest mean temporal CV MAE on development period; locked test opened once after selection/tuning.",
            }
        ]
    ).to_csv(paths.best / "10_final_locked_test_metrics.csv", index=False)

    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    fitted_model = best_pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        encoded_importance = pd.DataFrame(
            {
                "encoded_feature": feature_names,
                "importance": fitted_model.feature_importances_,
            }
        )
    elif hasattr(fitted_model, "coef_"):
        encoded_importance = pd.DataFrame(
            {
                "encoded_feature": feature_names,
                "importance": np.abs(np.ravel(fitted_model.coef_)),
            }
        )
    else:
        encoded_importance = pd.DataFrame(
            {"encoded_feature": feature_names, "importance": np.nan}
        )
    encoded_importance = encoded_importance.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    top3 = encoded_importance.head(3)
    LOGGER.info(
        "Top-3 encoded features: %s",
        ", ".join(f"{r['encoded_feature']}={r['importance']:.4f}" for _, r in top3.iterrows()),
    )
    encoded_importance.to_csv(
        paths.best / "10_encoded_feature_importance.csv", index=False
    )

    LOGGER.info("Computing Stage 10 permutation importance")
    LOGGER.debug(
        "Permutation importance features=%d repeats=%d workers=all",
        len(MODEL_FEATURES),
        5,
    )
    permutation = permutation_importance(
        best_pipeline,
        locked[list(MODEL_FEATURES)],
        locked[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    raw_importance = (
        pd.DataFrame(
            {
                "raw_feature": MODEL_FEATURES,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    raw_importance.to_csv(
        paths.best / "10_raw_feature_permutation_importance.csv", index=False
    )
    plot_best_outputs(
        locked[TARGET],
        locked_predictions,
        encoded_importance,
        raw_importance,
        paths.best,
    )

    caveats = {
        "two_feature_concentration": (
            "job_category and years_of_experience account for over 92% of total "
            "feature importance. The model is essentially a two-feature model."
        ),
        "synthetic_signal_warning": (
            "years_of_experience carries a contradictory, likely-synthetic signal "
            "(Stage 6 finding). Its importance must be reported as model reliance, "
            "not causal salary evidence."
        ),
        "no_causal_claims": (
            "Correlation and importance values are diagnostic association, not "
            "causal proof. No real-world salary-driver claim is permitted from "
            "this dataset."
        ),
        "honest_r2_framing": (
            "The strong R-squared is honestly reported as 'good fit to this "
            "specific dataset', never as 'AI salaries are explained by...'."
        ),
    }
    caveats_path = paths.best / "10_interpretation_caveats.json"
    caveats_path.write_text(json.dumps(caveats, indent=2), encoding="utf-8")
    LOGGER.info("Interpretation caveats written to %s", caveats_path.name)

    out_of_fold_errors = out_of_fold_absolute_errors(dev, folds, selection)
    interval_q90 = float(np.quantile(out_of_fold_errors, 0.90))
    LOGGER.info("90th percentile prediction interval: ±%.0f USD", interval_q90)
    return (
        selected_name,
        best_pipeline,
        locked_predictions,
        final_metrics,
        fitted_preprocessor,
        selection,
        interval_q90,
    )
