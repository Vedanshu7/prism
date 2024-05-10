from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from prism.models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth=None, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class GradientBoostingModel(BaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 42):
        super().__init__(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class LogisticRegressionModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        super().__init__(random_state=random_state, C=C, max_iter=max_iter)
        self._model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class SVMClassifierModel(BaseModel):
    def __init__(self, C: float = 1.0, kernel: str = "rbf", probability: bool = True, random_state: int = 42):
        super().__init__(random_state=random_state, C=C, kernel=kernel)
        self._model = SVC(C=C, kernel=kernel, probability=probability, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVMClassifierModel":
        self.feature_names_ = X.columns.tolist()
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
