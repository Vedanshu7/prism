from __future__ import annotations
from typing import Type

from prism.models.base import BaseModel
from prism.models.classifiers import (
    RandomForestModel, GradientBoostingModel, LogisticRegressionModel, SVMClassifierModel
)
from prism.models.regressors import (
    RandomForestRegressorModel, GradientBoostingRegressorModel,
    LinearRegressionModel, ElasticNetModel,
)

_REGISTRY: dict[str, Type[BaseModel]] = {
    "random_forest":            RandomForestModel,
    "gradient_boosting":        GradientBoostingModel,
    "logistic_regression":      LogisticRegressionModel,
    "svm":                      SVMClassifierModel,
    "random_forest_regressor":  RandomForestRegressorModel,
    "gradient_boosting_regressor": GradientBoostingRegressorModel,
    "linear_regression":        LinearRegressionModel,
    "elastic_net":              ElasticNetModel,
}


def register(name: str, cls: Type[BaseModel]) -> None:
    _REGISTRY[name] = cls


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: '{name}'. Available: {list_available()}")
    return _REGISTRY[name]


def list_available() -> list[str]:
    return sorted(_REGISTRY.keys())
