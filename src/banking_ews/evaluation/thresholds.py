"""Decision-threshold utilities for crisis early warning scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score


def candidate_thresholds(scores) -> np.ndarray:
    """Return stable candidate thresholds spanning observed model scores."""
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.5])
    unique = np.unique(arr)
    return np.r_[0.0, unique, 1.0]


def threshold_metrics(y_true, scores, threshold: float, beta: float = 2.0) -> dict[str, float]:
    """Compute precision, recall, and F-beta at a fixed alert threshold."""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f_beta": float(fbeta_score(y, pred, beta=beta, zero_division=0)),
        "alert_rate": float(pred.mean()) if pred.size else 0.0,
    }


def select_threshold(y_true, scores, beta: float = 2.0) -> dict[str, float]:
    """
    Pick the threshold with the highest F-beta score.

    beta=2 is the project default because missing a crisis is usually more
    costly than flagging an extra country-year for review.
    """
    rows = [threshold_metrics(y_true, scores, t, beta=beta) for t in candidate_thresholds(scores)]
    table = pd.DataFrame(rows).sort_values(
        ["f_beta", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
    )
    return table.iloc[0].to_dict()
