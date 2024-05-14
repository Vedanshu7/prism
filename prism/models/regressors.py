from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet

from prism.models.base import BaseModel


class RandomForestRegressorModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth=None, random_state: int = 42):
        super().__init__(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
        self._model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRegressorModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X):
        return None


class GradientBoostingRegressorModel(BaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate)
        self._model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingRegressorModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X):
        return None


class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X):
        return None


class ElasticNetModel(BaseModel):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, random_state: int = 42):
        super().__init__(random_state=random_state, alpha=alpha, l1_ratio=l1_ratio)
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X):
        return None
