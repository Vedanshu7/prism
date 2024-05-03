from __future__ import annotations
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_strategy: str = "mean",
    scale: bool = True,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]

    transformers = [
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", Pipeline(cat_steps),     categorical_cols),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")
