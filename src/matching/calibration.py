"""
calibration.py — Probability Calibration and Reliability Metrics.

Provides:
- Brier score computation
- Expected Calibration Error (ECE) and reliability curve statistics
- CalibratedModel wrapper (Platt scaling / Isotonic regression)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of predicted probabilities):
    Brier = (1/N) * sum((y_prob - y_true)^2)
    Lower is better (0.0 is perfect calibration and sharpness).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration curve and Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    N = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    bin_empirical_pos: list[float] = []
    bin_mean_preds: list[float] = []
    bin_counts: list[int] = []

    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            empirical_pos = float(np.mean(y_true[mask]))
            ece += (count / N) * abs(mean_pred - empirical_pos)
        else:
            mean_pred = (low + high) / 2.0
            empirical_pos = 0.0

        bin_centers.append((low + high) / 2.0)
        bin_mean_preds.append(mean_pred)
        bin_empirical_pos.append(empirical_pos)
        bin_counts.append(count)

    return {
        "brier_score": brier_score(y_true, y_prob),
        "ece": float(ece),
        "bin_centers": bin_centers,
        "bin_mean_preds": bin_mean_preds,
        "bin_empirical_pos": bin_empirical_pos,
        "bin_counts": bin_counts,
    }


class CalibratedModel:
    """
    Calibrates predicted probabilities of a trained base model using Platt scaling (sigmoid)
    or Isotonic Regression fitted on a validation/holdout set.
    """

    def __init__(self, base_model: Any, method: str = "sigmoid") -> None:
        self.base_model = base_model
        self.method = method
        self.calibrator: Any = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> CalibratedModel:
        """
        Fit the calibrator on holdout (X, y) predictions.
        """
        raw_probs = self.base_model.predict_proba(X)
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_probs, y)
        elif self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(raw_probs.reshape(-1, 1), y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted.")
        raw_probs = self.base_model.predict_proba(X)
        if self.method == "isotonic":
            return self.calibrator.predict(raw_probs)
        else:
            return self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)
