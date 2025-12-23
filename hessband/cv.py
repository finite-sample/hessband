"""
Cross-validation utilities for kernel regression and density estimation.

This module defines a CVScorer class that can be used to evaluate
leave-one-out cross-validation (LOOCV) or K-fold cross-validation
for kernel regression or density estimation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.model_selection import KFold  # type: ignore

__all__ = ["CVScorer"]


class CVScorer:
    """Cross-validation scorer for kernel regression.

    Args:
        X: Input values.
        y: Target values.
        folds: Number of folds for K-fold cross-validation.
        kernel: Kernel type ('gaussian' or 'epanechnikov').
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, folds: int = 5, kernel: str = "gaussian"
    ) -> None:
        self.X = np.asarray(X).ravel()
        self.y = np.asarray(y).ravel()
        if not (2 <= folds <= len(self.X)):
            raise ValueError(
                f"`folds` must be between 2 and {len(self.X)}, got {folds}"
            )
        self.kf = KFold(n_splits=folds, shuffle=True, random_state=0)
        self.kernel = kernel
        self.evals = 0

    def score(
        self,
        predict_fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray
        ],
        h: float,
    ) -> float:
        """Computes the cross-validation MSE for a given bandwidth.

        Args:
            predict_fn: Function that takes ``(X_train, y_train, X_test, h,
                kernel)`` and returns predictions.
            h: Bandwidth value.

        Returns:
            Cross-validation mean squared error.
        """
        mses = []
        for train_idx, test_idx in self.kf.split(self.X):
            Xtr, Xte = self.X[train_idx], self.X[test_idx]
            ytr, yte = self.y[train_idx], self.y[test_idx]
            ypred = predict_fn(Xtr, ytr, Xte, h, self.kernel)
            mses.append(mean_squared_error(yte, ypred))
            self.evals += 1
        return float(np.mean(mses))
