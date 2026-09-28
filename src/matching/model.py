"""
model.py — Supervised Pairwise Matching Models (Logistic Regression & LightGBM).

Lightweight tabular models designed for pairwise classification:
1. PairwiseLogisticRegressionMatcher: linear baseline with feature scaling.
2. PairwiseLightGBMMatcher: gradient-boosted tree matcher with high-order feature interactions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class PairwiseLogisticRegressionMatcher:
    """
    Logistic Regression pairwise matcher with standard scaling.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> PairwiseLogisticRegressionMatcher:
        """Fit scaler and logistic regression model."""
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1D array of positive class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at specified decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return absolute coefficient magnitudes."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return np.abs(self.model.coef_[0])

    def save(self, filepath: str | Path) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> PairwiseLogisticRegressionMatcher:
        with open(filepath, "rb") as f:
            return pickle.load(f)


class PairwiseLightGBMMatcher:
    """
    Gradient-boosted decision tree matcher using LightGBM.
    Captures non-linear feature interactions (name x address x provenance).
    """

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if not _HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for PairwiseLightGBMMatcher.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> PairwiseLightGBMMatcher:
        """Fit LightGBM model."""
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=25, verbose=False)]

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1D array of positive class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at specified decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return split-based feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.feature_importances_

    def save(self, filepath: str | Path) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> PairwiseLightGBMMatcher:
        with open(filepath, "rb") as f:
            return pickle.load(f)
