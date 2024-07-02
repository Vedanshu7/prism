import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":    np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n),
        "score":  np.random.uniform(0, 1, n),
        "city":   np.random.choice(["Mumbai", "Delhi", "Pune"], n),
        "target": np.random.choice(["yes", "no"], n),
    })


@pytest.fixture
def sample_regression_df():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.1
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    return df


@pytest.fixture
def X_train_clf(sample_classification_df):
    return sample_classification_df.drop(columns=["target", "city"])


@pytest.fixture
def y_train_clf(sample_classification_df):
    return sample_classification_df["target"]
