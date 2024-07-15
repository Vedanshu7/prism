import os
import pickle
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def saved_model(tmp_path):
    from sklearn.datasets import make_classification
    import pandas as pd
    from prism.models.classifiers import RandomForestModel

    X, y = make_classification(n_samples=50, n_features=3, random_state=42)
    X_df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    model = RandomForestModel(n_estimators=5, random_state=42)
    model.fit(X_df, pd.Series(y))

    path = tmp_path / "model.pkl"
    model.save(str(path))
    return str(path)


@pytest.fixture
def client(saved_model):
    os.environ["MODEL_PATH"] = saved_model
    from prism.api.app import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model" in data


def test_predict(client):
    resp = client.post("/predict", json={"features": {"f0": 0.5, "f1": -0.3, "f2": 1.2}})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
