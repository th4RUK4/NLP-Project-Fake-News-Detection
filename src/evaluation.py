"""Evaluation helpers shared by the model notebooks."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true: Any, y_pred: Any, y_prob: Any | None = None) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
    return result


def append_metrics(metrics: dict[str, Any], path: str) -> pd.DataFrame:
    row = pd.DataFrame([metrics])
    try:
        existing = pd.read_csv(path)
        result = pd.concat([existing, row], ignore_index=True)
    except FileNotFoundError:
        result = row
    result.to_csv(path, index=False)
    return result
