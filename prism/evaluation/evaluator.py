from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from prism.evaluation.metrics import classification_metrics, regression_metrics
from prism.models.base import BaseModel

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, task: str = "classification"):
        self.task = task
        self.results_: dict[str, Any] = {}

    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        if self.task == "classification":
            metrics = classification_metrics(y, y_pred, y_proba)
        else:
            metrics = regression_metrics(y, y_pred)

        self.results_ = {
            "model":   model.__class__.__name__,
            "task":    self.task,
            "metrics": metrics,
            "params":  model.get_params(),
        }
        logger.info(f"Evaluation: {metrics}")
        return self.results_

    def compare_models(self, models: dict[str, BaseModel], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        rows = []
        for name, model in models.items():
            result = self.evaluate(model, X, y)
            row = {"model": name}
            row.update(result["metrics"])
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        logger.info(f"Model comparison:\n{df}")
        return df

    def plot_feature_importance(self, model: BaseModel, output_path: Optional[str] = None) -> None:
        importances = model.feature_importances()
        if not importances:
            logger.warning("Model has no feature importances")
            return
        try:
            import matplotlib.pyplot as plt
            items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
            names, vals = zip(*items)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(names)), vals)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances")
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available for plots")
