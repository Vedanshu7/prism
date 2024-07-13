import pandas as pd
import pytest
from sklearn.datasets import make_classification
from prism.models.classifiers import RandomForestModel
from prism.evaluation.evaluator import Evaluator
from prism.evaluation.metrics import classification_metrics, regression_metrics
import numpy as np


@pytest.fixture
def trained_model():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df, y_ser = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X_df, y_ser)
    return model, X_df, y_ser


def test_evaluate_classification(trained_model):
    model, X, y = trained_model
    evaluator = Evaluator(task="classification")
    results = evaluator.evaluate(model, X, y)
    assert "metrics" in results
    assert "accuracy" in results["metrics"]
    assert 0 <= results["metrics"]["accuracy"] <= 1


def test_classification_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    metrics = classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    metrics = regression_metrics(y_true, y_pred)
    assert "mse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.9
