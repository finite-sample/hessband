"""
Cross-validation utilities for kernel regression and density estimation.

This module defines a CVScorer class that can be used to evaluate
leave-one-out cross-validation (LOOCV) or K-fold cross-validation
for kernel regression or density estimation.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

__all__ = ["CVScorer"]

class CVScorer:
    """
    Cross-validation scorer for kernel regression.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Input values.
    y : array-like, shape (n_samples,)
        Target values.
    folds : int, optional (default=5)
        Number of folds for K-fold cross-validation.
    kernel : str, optional (default='gaussian')
        Kernel type ('gaussian' or 'epanechnikov').
    """
    def __init__(self, X, y, folds=5, kernel='gaussian'):
        self.X = np.asarray(X).ravel()
        self.y = np.asarray(y).ravel()
        if not (2 <= folds <= len(self.X)):
            raise ValueError(
                f"`folds` must be between 2 and {len(self.X)}, got {folds}"
            )
        self.kf = KFold(n_splits=folds, shuffle=True, random_state=0)
        self.kernel = kernel
        self.evals = 0

    def score(self, predict_fn, h):
        """
        Compute the cross-validation MSE for a given bandwidth.

        Parameters
        ----------
        predict_fn : callable
            Function that takes ``(X_train, y_train, X_test, h, kernel)`` and
            returns predictions.
        h : float
            Bandwidth value.

        Returns
        -------
        float
            Cross-validation mean squared error.
        """
        mses = []
        for train_idx, test_idx in self.kf.split(self.X):
            Xtr, Xte = self.X[train_idx], self.X[test_idx]
            ytr, yte = self.y[train_idx], self.y[test_idx]
            ypred = predict_fn(Xtr, ytr, Xte, h, kernel=self.kernel)
            mses.append(mean_squared_error(yte, ypred))
            self.evals += 1
        return np.mean(mses)
