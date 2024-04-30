from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, permutation_importance
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features_: list[str] = []

    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        estimator=None,
    ) -> list[str]:
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, len(X.columns)))
        rfe.fit(X, y)
        self.selected_features_ = X.columns[rfe.support_].tolist()
        logger.info(f"RFE selected {len(self.selected_features_)} features")
        return self.selected_features_

    def permutation_importance_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator=None,
        threshold: float = 0.0,
        n_repeats: int = 5,
    ) -> list[str]:
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        estimator.fit(X, y)
        result = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=self.random_state)
        self.selected_features_ = [
            col for col, imp in zip(X.columns, result.importances_mean) if imp > threshold
        ]
        return self.selected_features_

    def shap_values(self, model, X: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            return explainer.shap_values(X)
        except Exception as e:
            logger.warning(f"SHAP not available: {e}")
            return None
