from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.constants import SEED


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    factory: Callable[[], Any]
    scale_numeric: bool
    is_baseline: bool = False


def model_catalog() -> list[ModelDefinition]:
    return [
        ModelDefinition(
            "Dummy Median",
            lambda: DummyRegressor(strategy="median"),
            False,
            True,
        ),
        ModelDefinition("Linear Regression", LinearRegression, True),
        ModelDefinition("Ridge Regression", lambda: Ridge(alpha=10.0), True),
        ModelDefinition(
            "Random Forest",
            lambda: RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=SEED,
                n_jobs=1,
            ),
            False,
        ),
        ModelDefinition(
            "Gradient Boosting",
            lambda: GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=2,
                loss="huber",
                random_state=SEED,
            ),
            False,
        ),
    ]
