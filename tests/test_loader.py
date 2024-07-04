import os, tempfile
import numpy as np
import pandas as pd
import pytest
from prism.data.loader import DataLoader


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6], "target": ["x","y","x"]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_csv(csv_file):
    loader = DataLoader()
    df = loader.load(csv_file)
    assert len(df) == 3
    assert set(df.columns) == {"a", "b", "target"}


def test_load_missing_file():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/file.csv")


def test_validate_schema(csv_file):
    loader = DataLoader()
    df = loader.load(csv_file)
    loader.validate_schema(df, ["a", "b"])
    with pytest.raises(ValueError):
        loader.validate_schema(df, ["nonexistent_col"])


def test_split_train_test(csv_file):
    loader = DataLoader(random_state=42)
    df = pd.DataFrame({
        "a": range(100), "b": range(100),
        "target": ["yes" if i % 2 == 0 else "no" for i in range(100)]
    })
    X_train, X_test, y_train, y_test = loader.split_train_test(df, "target", test_size=0.2)
    assert len(X_train) + len(X_test) == 100
    assert abs(len(X_test) - 20) <= 2
