from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def load(self, path: str) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
        return df

    def validate_schema(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def split_train_test(
        self, df: pd.DataFrame, target: str, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y if y.dtype == object else None)

    def load_split(
        self, path: str, target: str, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = self.load(path)
        return self.split_train_test(df, target, test_size)
