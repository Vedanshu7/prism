from __future__ import annotations
import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
)

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self):
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler: Optional[StandardScaler | MinMaxScaler] = None
        self._ohe: Optional[OneHotEncoder] = None
        self._ohe_columns: list[str] = []

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Literal["mean", "median", "mode", "drop"] = "mean",
        fill_value=None,
    ) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(f"Dropped {before - len(df)} rows with missing values")
            return df

        for col in numeric_cols:
            if df[col].isna().any():
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

        for col in cat_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else "unknown")

        return df

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        method: Literal["label", "onehot"] = "label",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        cat_cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()

        if method == "label":
            for col in cat_cols:
                if col not in self._label_encoders:
                    self._label_encoders[col] = LabelEncoder()
                    df[col] = self._label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self._label_encoders[col].transform(df[col].astype(str))
        elif method == "onehot":
            self._ohe_columns = cat_cols
            self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = self._ohe.fit_transform(df[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=self._ohe.get_feature_names_out(cat_cols), index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

        return df

    def normalize(
        self,
        df: pd.DataFrame,
        method: Literal["standard", "minmax"] = "standard",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        num_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = MinMaxScaler()

        df[num_cols] = self._scaler.fit_transform(df[num_cols])
        return df

    def fix_skew(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if abs(df[col].skew()) > threshold and df[col].min() > 0:
                df[col] = np.log1p(df[col])
                logger.debug(f"Applied log1p to skewed column: {col}")
        return df
