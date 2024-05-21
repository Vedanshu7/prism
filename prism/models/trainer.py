from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from prism.models.base import BaseModel
from prism.utils.io import save_pickle

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_model_: Optional[BaseModel] = None
        self.cv_scores_: dict[str, float] = {}

    def train(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        logger.info(f"Training {model.__class__.__name__} on {X.shape} data")
        model.fit(X, y)
        self.best_model_ = model
        return model

    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "f1_weighted",
    ) -> dict[str, float]:
        scores = cross_val_score(model._model, X, y, cv=cv, scoring=scoring)
        result = {
            "mean":   float(scores.mean()),
            "std":    float(scores.std()),
            "scores": scores.tolist(),
        }
        self.cv_scores_ = result
        logger.info(f"CV {scoring}: {result['mean']:.4f} ± {result['std']:.4f}")
        return result

    def hyperparameter_search(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, Any],
        method: str = "grid",
        cv: int = 5,
        n_iter: int = 20,
        scoring: str = "f1_weighted",
    ) -> BaseModel:
        if method == "grid":
            search = GridSearchCV(model._model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model._model, param_grid, n_iter=n_iter, cv=cv,
                                        scoring=scoring, random_state=self.random_state, n_jobs=-1)
        search.fit(X, y)
        logger.info(f"Best params: {search.best_params_}  score: {search.best_score_:.4f}")
        model._model = search.best_estimator_
        model.params.update(search.best_params_)
        return model

    def save_best_model(self, path: str | Path) -> None:
        if self.best_model_ is None:
            raise RuntimeError("No model trained yet")
        save_pickle(self.best_model_, path)
        logger.info(f"Saved model to {path}")
