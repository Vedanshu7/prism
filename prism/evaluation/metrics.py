from __future__ import annotations
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse":  float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def confusion_matrix_dict(y_true, y_pred) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    return {"matrix": cm.tolist(), "labels": sorted(set(y_true))}
