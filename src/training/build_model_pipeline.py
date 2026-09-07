from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline

from src.models import ModelDefinition
from src.training.build_pipeline import build_pipeline

LOGGER = logging.getLogger("ml_pipeline.training")


def build_model_pipeline(definition: ModelDefinition) -> Pipeline:
    """Build a fresh pipeline from a constructor-based model definition."""
    LOGGER.debug(
        "Building pipeline for %s scale_numeric=%s",
        definition.name,
        definition.scale_numeric,
    )
    return build_pipeline(definition.build_estimator(), definition.scale_numeric)


__all__ = ["build_model_pipeline"]
