from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from prism.utils.io import save_pickle, load_pickle


class BaseModel(ABC):
    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.params = kwargs
        self._model = None
        self.feature_names_: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def get_params(self) -> dict[str, Any]:
        return {"model": self.__class__.__name__, **self.params}

    def save(self, path: str | Path) -> None:
        save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return load_pickle(path)

    def feature_importances(self) -> Optional[dict[str, float]]:
        if hasattr(self._model, "feature_importances_") and self.feature_names_:
            return dict(zip(self.feature_names_, self._model.feature_importances_))
        return None
