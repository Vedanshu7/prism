import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from prism.models.classifiers import RandomForestModel, LogisticRegressionModel, GradientBoostingModel


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_ser = pd.Series(y)
    return X_df, y_ser


def test_random_forest_fit_predict(clf_data):
    X, y = clf_data
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_logistic_regression_fit_predict(clf_data):
    X, y = clf_data
    model = LogisticRegressionModel(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_predict_proba(clf_data):
    X, y = clf_data
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba is not None
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_feature_importances(clf_data):
    X, y = clf_data
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X, y)
    imp = model.feature_importances()
    assert imp is not None
    assert len(imp) == 10
    assert abs(sum(imp.values()) - 1.0) < 1e-5
