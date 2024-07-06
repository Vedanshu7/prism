import numpy as np
import pandas as pd
import pytest
from prism.data.preprocessor import Preprocessor


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "age":    [25, None, 35, 40, None],
        "income": [50000, 60000, None, 80000, 70000],
        "city":   ["Mumbai", None, "Delhi", "Pune", "Mumbai"],
    })


def test_handle_missing_mean(df_with_nulls):
    p = Preprocessor()
    result = p.handle_missing_values(df_with_nulls, strategy="mean")
    assert result.isnull().sum().sum() == 0
    assert result["age"].iloc[1] == pytest.approx(df_with_nulls["age"].mean(), rel=1e-3)


def test_handle_missing_drop():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    p = Preprocessor()
    result = p.handle_missing_values(df, strategy="drop")
    assert len(result) == 1


def test_encode_label():
    df = pd.DataFrame({"city": ["Mumbai", "Delhi", "Pune", "Mumbai"]})
    p = Preprocessor()
    result = p.encode_categoricals(df, method="label")
    assert result["city"].dtype in [int, np.int64, np.int32]
    assert len(result["city"].unique()) == 3


def test_normalize_standard():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    p = Preprocessor()
    result = p.normalize(df, method="standard")
    assert abs(result["a"].mean()) < 1e-10
    assert abs(result["a"].std() - 1.0) < 0.1
