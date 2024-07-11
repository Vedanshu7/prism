import pandas as pd
import pytest
from sklearn.datasets import make_classification
from prism.models.classifiers import RandomForestModel
from prism.models.trainer import Trainer


@pytest.fixture
def clf_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


def test_trainer_basic(clf_data):
    X, y = clf_data
    model = RandomForestModel(n_estimators=5, random_state=42)
    trainer = Trainer(random_state=42)
    trained = trainer.train(model, X, y)
    assert trained is model
    assert trainer.best_model_ is model


def test_trainer_cross_validate(clf_data):
    X, y = clf_data
    model = RandomForestModel(n_estimators=5, random_state=42)
    trainer = Trainer(random_state=42)
    trainer.train(model, X, y)
    scores = trainer.cross_validate(model, X, y, cv=3, scoring="accuracy")
    assert "mean" in scores
    assert "std" in scores
    assert 0 <= scores["mean"] <= 1


def test_trainer_save_load(clf_data, tmp_path):
    X, y = clf_data
    model = RandomForestModel(n_estimators=5, random_state=42)
    trainer = Trainer()
    trainer.train(model, X, y)
    path = tmp_path / "model.pkl"
    trainer.save_best_model(str(path))
    assert path.exists()

    from prism.models.base import BaseModel
    loaded = BaseModel.load(str(path))
    preds = loaded.predict(X)
    assert len(preds) == len(y)
