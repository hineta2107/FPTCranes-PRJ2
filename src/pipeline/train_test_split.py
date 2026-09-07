"""Stage 8: Train-Test Split."""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from src.constants import LOCKED_MONTH, LOCKED_YEAR, MODEL_FEATURES, TARGET
from src.models import PipelinePaths
from src.training import build_preprocessor

LOGGER = logging.getLogger("ml_pipeline")


def train_test_split(
    clean: pd.DataFrame,
    dev: pd.DataFrame,
    locked: pd.DataFrame,
    paths: PipelinePaths,
) -> None:
    pd.DataFrame(
        [
            {
                "partition": "TRAIN_DEV",
                "rows": len(dev),
                "pct": len(dev) / len(clean) * 100,
                "period": f"Before {LOCKED_YEAR}-{LOCKED_MONTH:02d}",
            },
            {
                "partition": "LOCKED_TEST",
                "rows": len(locked),
                "pct": len(locked) / len(clean) * 100,
                "period": f"{LOCKED_YEAR}-{LOCKED_MONTH:02d}",
            },
        ]
    ).to_csv(paths.ml_ready / "08_split_summary.csv", index=False)
    LOGGER.info(
        "Temporal split prepared: development=%d locked_test=%d", len(dev), len(locked)
    )
    LOGGER.debug(
        "Split on raw cleaned df (not df_corr), total=%d", len(clean)
    )
    dev[list(MODEL_FEATURES) + [TARGET, "posting_year", "posting_month"]].to_csv(
        paths.ml_ready / "train_raw_model_input.csv", index=False
    )
    locked[list(MODEL_FEATURES) + ["posting_year", "posting_month"]].to_csv(
        paths.ml_ready / "locked_test_raw_features.csv", index=False
    )
    locked[[TARGET]].to_csv(
        paths.ml_ready / "LOCKED_TEST_TARGET_FINAL_EVAL_ONLY.csv", index=False
    )

    preprocessor = build_preprocessor(scale_numeric=True)
    LOGGER.info("Fitting preprocessor on TRAIN partition only (%d rows)", len(dev))
    x_train = preprocessor.fit_transform(dev[list(MODEL_FEATURES)])
    x_test = preprocessor.transform(locked[list(MODEL_FEATURES)])
    feature_names = preprocessor.get_feature_names_out()
    LOGGER.info("Encoded feature count: %d columns", len(feature_names))
    LOGGER.debug("X_train shape=%s X_test shape=%s", x_train.shape, x_test.shape)
    pd.DataFrame({"encoded_feature": feature_names}).to_csv(
        paths.ml_ready / "08_encoded_feature_names.csv", index=False
    )
    train_ready = pd.DataFrame(x_train, columns=feature_names)
    train_ready[TARGET] = dev[TARGET].to_numpy()
    test_ready = pd.DataFrame(x_test, columns=feature_names)
    test_ready[TARGET] = locked[TARGET].to_numpy()
    train_ready.to_csv(
        paths.ml_ready / "data_ready_for_machine_learning_TRAIN.csv", index=False
    )
    test_ready.to_csv(
        paths.ml_ready / "data_ready_for_machine_learning_LOCKED_TEST_FINAL_EVAL.csv",
        index=False,
    )
    joblib.dump(preprocessor, paths.artifacts / "preprocessor_ml_ready.joblib")
