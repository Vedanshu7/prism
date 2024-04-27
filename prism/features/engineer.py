from __future__ import annotations
import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def create_interaction_features(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None, max_pairs: int = 10
    ) -> pd.DataFrame:
        df = df.copy()
        num_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        pairs = list(combinations(num_cols[:max_pairs], 2))
        for c1, c2 in pairs:
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
        logger.info(f"Created {len(pairs)} interaction features")
        return df

    def create_polynomial_features(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None, degree: int = 2
    ) -> pd.DataFrame:
        df = df.copy()
        num_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            for d in range(2, degree + 1):
                df[f"{col}_pow{d}"] = df[col] ** d
        return df

    def select_k_best(
        self, X: pd.DataFrame, y: pd.Series, k: int = 10, score_func=f_classif
    ) -> pd.DataFrame:
        selector = SelectKBest(score_func=score_func, k=min(k, len(X.columns)))
        selector.fit(X, y)
        selected = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected)} best features")
        return X[selected]

    def select_by_variance(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        selected = df.columns[selector.get_support()].tolist()
        logger.info(f"Kept {len(selected)}/{len(df.columns)} features after variance filter")
        return df[selected]

    def drop_high_correlation(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        df = df.copy()
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        return df.drop(columns=to_drop)
