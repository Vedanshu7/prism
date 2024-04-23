from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._issues: list[str] = []

    def check_shape(self, min_rows: int = 10, min_cols: int = 1) -> "DataValidator":
        r, c = self.df.shape
        if r < min_rows:
            self._issues.append(f"Too few rows: {r} (min {min_rows})")
        if c < min_cols:
            self._issues.append(f"Too few columns: {c} (min {min_cols})")
        return self

    def check_nulls(self, threshold: float = 0.5) -> "DataValidator":
        null_pct = self.df.isnull().mean()
        high_null = null_pct[null_pct > threshold]
        if not high_null.empty:
            self._issues.append(f"High null columns (>{threshold*100:.0f}%): {high_null.index.tolist()}")
        return self

    def check_dtypes(self) -> "DataValidator":
        object_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            pass  # informational only
        return self

    def check_target_balance(self, target: str, min_class_ratio: float = 0.05) -> "DataValidator":
        if target not in self.df.columns:
            self._issues.append(f"Target column '{target}' not found")
            return self
        counts = self.df[target].value_counts(normalize=True)
        rare = counts[counts < min_class_ratio]
        if not rare.empty:
            self._issues.append(f"Rare classes (<{min_class_ratio*100:.0f}%): {rare.index.tolist()}")
        return self

    def generate_report(self) -> dict[str, Any]:
        df = self.df
        return {
            "rows":            len(df),
            "columns":         len(df.columns),
            "null_counts":     df.isnull().sum().to_dict(),
            "null_pct":        (df.isnull().mean() * 100).round(2).to_dict(),
            "dtypes":          {c: str(t) for c, t in df.dtypes.items()},
            "numeric_stats":   df.describe().to_dict(),
            "issues":          self._issues,
            "passed":          len(self._issues) == 0,
        }

    def raise_if_invalid(self) -> None:
        if self._issues:
            raise ValueError("Data validation failed:\n" + "\n".join(f"  - {i}" for i in self._issues))
