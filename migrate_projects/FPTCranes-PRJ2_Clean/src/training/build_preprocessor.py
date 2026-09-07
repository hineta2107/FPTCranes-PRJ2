from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.constants import COMPANY_SIZE_ORDER, EDUCATION_ORDER


LOGGER = logging.getLogger("ml_pipeline.training")


# Keep the feature-candidate structure from build_preprocessor.py.
NOMINAL_CANDIDATES = [
    "job_title",
    "job_category",
    "city",
    "country",
    "remote_work",
    "industry",
]

NUMERIC_CANDIDATES = [
    "years_of_experience",
    "demand_score",
    "benefits_score_10",
    "skill_count",
]

LEGACY_COLUMN_RENAME = {
    "AI Engineering": "job_category",
}


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Optional compatibility transformer retained from build_preprocessor12.py.

    It is intentionally NOT inserted inside build_preprocessor(), because the
    original build_preprocessor.py contract returns a ColumnTransformer and is
    already driven by canonical feature names.

    Use build_preprocessor_pipeline(...) only when raw input may still contain
    legacy names such as "AI Engineering".
    """

    def __init__(self, rename_mapping: dict[str, str]):
        self.rename_mapping = rename_mapping

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnRenamer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ColumnRenamer expects a pandas DataFrame so column names can be normalized."
            )
        return X.rename(columns=self.rename_mapping)


def one_hot_encoder() -> OneHotEncoder:
    """
    Scikit-learn compatible OneHotEncoder.

    Keep the local helper from build_preprocessor.py so this module does not
    depend on a second helper file and still supports sklearn < 1.2.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    feature_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Construct an unfitted, leakage-safe feature preprocessor.

    Structure intentionally follows build_preprocessor.py:
      - dynamic selection from feature_columns
      - nominal OHE with most-frequent imputation
      - ordered education and company-size encoding
      - experience_level only when explicitly requested for ablation
      - median numeric imputation + missing indicators
      - optional RobustScaler
      - binary multi-hot skill tokenization
      - ColumnTransformer return type
    """
    LOGGER.debug(
        "Building preprocessor with %d requested columns; scale_numeric=%s",
        len(feature_columns),
        scale_numeric,
    )

    transformers: list[tuple[str, Any, Any]] = []

    nominal = [c for c in NOMINAL_CANDIDATES if c in feature_columns]
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot_encoder()),
                    ]
                ),
                nominal,
            )
        )

    if "education_required" in feature_columns:
        education = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[list(EDUCATION_ORDER)],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("education", education, ["education_required"]))

    if "company_size" in feature_columns:
        company = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[list(COMPANY_SIZE_ORDER)],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("company_size", company, ["company_size"]))

    # experience_level is used only in the explicit A2 ablation experiment.
    if "experience_level" in feature_columns:
        experience = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[
                            [
                                "Entry (0-2 yrs)",
                                "Mid (3-5 yrs)",
                                "Senior (6-9 yrs)",
                                "Lead (10+ yrs)",
                            ]
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("experience_level", experience, ["experience_level"]))

    numeric = [c for c in NUMERIC_CANDIDATES if c in feature_columns]
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            )
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", RobustScaler()))

        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric,
            )
        )

    if "required_skills" in feature_columns:
        skills = CountVectorizer(
            token_pattern=r"[^|]+",
            lowercase=False,
            binary=True,
        )
        transformers.append(("skills", skills, "required_skills"))

    LOGGER.debug(
        "Configured transformer families: %s",
        [name for name, _, _ in transformers],
    )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def build_preprocessor_pipeline(
    feature_columns: list[str],
    scale_numeric: bool,
    rename_legacy_columns: bool = True,
) -> Pipeline:
    """
    Optional wrapper combining the legacy-column renaming idea from
    build_preprocessor12.py with the canonical build_preprocessor().

    The main build_preprocessor() function remains unchanged in structure and
    return type for backwards compatibility.
    """
    steps: list[tuple[str, Any]] = []

    if rename_legacy_columns:
        steps.append(
            (
                "rename_legacy_columns",
                ColumnRenamer(LEGACY_COLUMN_RENAME),
            )
        )

    steps.append(
        (
            "features",
            build_preprocessor(
                feature_columns=feature_columns,
                scale_numeric=scale_numeric,
            ),
        )
    )

    return Pipeline(steps=steps)


__all__ = [
    "build_preprocessor",
    "build_preprocessor_pipeline",
    "one_hot_encoder",
    "ColumnRenamer",
    "NOMINAL_CANDIDATES",
    "NUMERIC_CANDIDATES",
]
