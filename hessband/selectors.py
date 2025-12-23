from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from math import erf
from typing import cast

import numpy as np
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import ConstantKernel as C  # type: ignore
from sklearn.gaussian_process.kernels import Matern, WhiteKernel  # type: ignore

from .cv import CVScorer
from .kernels import kernel_derivatives, kernel_weights

__all__ = [
    "select_nw_bandwidth",
    "nw_predict",
    "grid_search_cv",
    "plug_in_bandwidth",
    "newton_fd",
    "analytic_newton",
    "golden_section",
    "bayes_opt_bandwidth",
]

# ----------------------------------------------------------------------------
# Nadaraya–Watson prediction
# ----------------------------------------------------------------------------


def nw_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    h: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Computes Nadaraya-Watson predictions.

    Args:
        X_train: Training input values.
        y_train: Training target values.
        X_test: Test input values.
        h: Bandwidth.
        kernel: Kernel to use ('gaussian' or 'epanechnikov').

    Returns:
        The predicted values for X_test.
    """
    X_train = np.asarray(X_train).ravel()
    X_test = np.asarray(X_test).ravel()
    u = (X_test[:, None] - X_train[None, :]) / h
    w = kernel_weights(u, h, kernel)
    numerator = (w * y_train).sum(axis=1)
    denom = w.sum(axis=1)
    zero_mask = denom == 0
    if np.any(zero_mask):
        denom[zero_mask] = len(X_train)
        numerator[zero_mask] = y_train.sum()
    return cast(np.ndarray, numerator / denom)


# ----------------------------------------------------------------------------
# Basic bandwidth selectors
# ----------------------------------------------------------------------------


def grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray],
    h_grid: np.ndarray,
    folds: int = 5,
) -> float:
    """Performs grid search for bandwidth selection.

    Args:
        X: Input values.
        y: Target values.
        kernel: Kernel to use.
        predict_fn: Prediction function.
        h_grid: Grid of bandwidths to search over.
        folds: Number of folds for cross-validation.

    Returns:
        The best bandwidth found.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    best_h, best_score = None, np.inf
    for h in h_grid:
        score = scorer.score(predict_fn, h)
        if score < best_score:
            best_score, best_h = score, h
    assert best_h is not None
    return cast(float, best_h)


def plug_in_bandwidth(X: np.ndarray) -> float:
    """Computes a plug-in bandwidth.

    Uses Silverman's rule of thumb.

    Args:
        X: Input values.

    Returns:
        The plug-in bandwidth.
    """
    sigma = np.std(np.asarray(X).ravel(), ddof=1)
    n = len(X)
    return cast(float, 1.06 * sigma * n ** (-1 / 5))


def newton_fd(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray],
    h_init: float,
    h_min: float = 1e-3,
    folds: int = 5,
    tol: float = 1e-3,
    max_iter: int = 10,
    eps: float = 1e-4,
) -> float:
    """Finite-difference Newton method for bandwidth selection.

    Args:
        X: Input values.
        y: Target values.
        kernel: Kernel to use.
        predict_fn: Prediction function.
        h_init: Initial bandwidth.
        h_min: Minimum bandwidth.
        folds: Number of folds for cross-validation.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        eps: Epsilon for finite differences.

    Returns:
        The optimal bandwidth.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    h = max(h_init, h_min)
    for _ in range(max_iter):
        s0 = scorer.score(predict_fn, h)
        s1 = scorer.score(predict_fn, h + eps)
        s_1 = scorer.score(predict_fn, max(h - eps, h_min))
        grad = (s1 - s_1) / (2 * eps)
        hess = (s1 + s_1 - 2 * s0) / (eps**2)
        if hess <= 0:
            break
        h_new = max(h_min, h - grad / hess)
        if abs(h_new - h) < tol:
            h = h_new
            break
        h = h_new
    return h


def analytic_newton(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray],
    h_init: float,
    h_min: float = 1e-3,
    folds: int = 5,
    tol: float = 1e-3,
    max_iter: int = 10,
) -> float:
    """Analytic Newton method for LOOCV risk minimization.

    Returns the bandwidth without performing CV evaluations in the loop.

    Args:
        X: Input values.
        y: Target values.
        kernel: Kernel to use.
        predict_fn: Prediction function.
        h_init: Initial bandwidth.
        h_min: Minimum bandwidth.
        folds: Number of folds for cross-validation.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        The optimal bandwidth.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)

    def obj_grad_hess(h: float) -> tuple[float, float, float]:
        grad, hess, obj = 0.0, 0.0, 0.0
        total = 0
        for train_idx, test_idx in scorer.kf.split(scorer.X):
            Xtr, Xte = scorer.X[train_idx], scorer.X[test_idx]
            ytr, yte = scorer.y[train_idx], scorer.y[test_idx]
            u = (Xte[:, None] - Xtr[None, :]) / h
            w, d_w, dd_w = kernel_derivatives(u, h, kernel)
            w_sum = w.sum(axis=1)
            num = (w * ytr).sum(axis=1)
            zero_mask = w_sum == 0
            if np.any(zero_mask):
                w_sum[zero_mask] = len(ytr)
                num[zero_mask] = ytr.mean() * w_sum[zero_mask]
            m = num / w_sum
            residual = yte - m
            obj += np.sum(residual**2)
            d_num = (d_w * ytr).sum(axis=1)
            dd_num = (dd_w * ytr).sum(axis=1)
            d_den = d_w.sum(axis=1)
            dd_den = dd_w.sum(axis=1)
            dm = (d_num * w_sum - num * d_den) / (w_sum**2)
            ddm = (
                dd_num * w_sum
                - 2 * d_num * d_den
                - num * dd_den
                + 2 * num * (d_den**2) / w_sum
            ) / (w_sum**2)
            dm[zero_mask] = 0
            ddm[zero_mask] = 0
            grad += -2 * np.sum(residual * dm)
            hess += 2 * np.sum(dm**2 - residual * ddm)
            total += len(yte)
        return obj / total, grad, hess

    h = max(h_init, h_min)
    for _ in range(max_iter):
        obj_val, grad, hess = obj_grad_hess(h)
        direction = -grad / hess if hess > 0 and np.isfinite(hess) else -grad
        # Armijo line search
        c1, tau = 1e-4, 0.5
        alpha = 1.0
        while alpha > 1e-4:
            h_trial = max(h_min, h + alpha * direction)
            new_obj, _, _ = obj_grad_hess(h_trial)
            if new_obj <= obj_val + c1 * alpha * grad * direction:
                break
            alpha *= tau
        h_new = h_trial
        if abs(h_new - h) < tol:
            h = h_new
            break
        h = h_new
    return h


def golden_section(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray],
    a: float,
    b: float,
    folds: int = 5,
    tol: float = 1e-3,
    max_iter: int = 20,
) -> float:
    """Golden-section search for bandwidth selection.

    Args:
        X: Input values.
        y: Target values.
        kernel: Kernel to use.
        predict_fn: Prediction function.
        a: Lower bound of the search interval.
        b: Upper bound of the search interval.
        folds: Number of folds for cross-validation.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        The optimal bandwidth.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    phi = (1 + np.sqrt(5)) / 2
    c, d = b - (b - a) / phi, a + (b - a) / phi
    f_c, f_d = scorer.score(predict_fn, c), scorer.score(predict_fn, d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f_c < f_d:
            b, f_d = d, f_c
            d = c
            c = b - (b - a) / phi
            f_c = scorer.score(predict_fn, c)
        else:
            a, f_c = c, f_d
            c = d
            d = a + (b - a) / phi
            f_d = scorer.score(predict_fn, d)
    return (a + b) / 2


def bayes_opt_bandwidth(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, str], np.ndarray],
    a: float,
    b: float,
    folds: int = 5,
    init_points: int = 5,
    n_iter: int = 10,
) -> float:
    """Bayesian optimization for bandwidth selection.

    Args:
        X: Input values.
        y: Target values.
        kernel: Kernel to use.
        predict_fn: Prediction function.
        a: Lower bound of the search interval.
        b: Upper bound of the search interval.
        folds: Number of folds for cross-validation.
        init_points: Number of initial points for Bayesian optimization.
        n_iter: Number of iterations for Bayesian optimization.

    Returns:
        The optimal bandwidth.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    # Initial design points
    Xs = np.linspace(a, b, init_points)
    Ys = [scorer.score(predict_fn, x) for x in Xs]
    for _ in range(n_iter):
        X_train = Xs.reshape(-1, 1)
        y_train = np.array(Ys)
        base_kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
        wk = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e6))
        kernel_gp = base_kernel + wk
        attempts = 0
        while attempts < 2:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gp = GaussianProcessRegressor(
                    kernel=kernel_gp,
                    normalize_y=True,
                ).fit(X_train, y_train)
                # If noise_level hits lower bound, relax it
                if any(issubclass(wi.category, ConvergenceWarning) for wi in w):
                    lb, ub = wk.noise_level_bounds
                    new_lb = max(lb / 10.0, 1e-8)
                    wk = WhiteKernel(
                        noise_level=wk.noise_level,
                        noise_level_bounds=(new_lb, ub),
                    )
                    kernel_gp = base_kernel + wk
                    attempts += 1
                    continue
                break
        hs = np.linspace(a, b, 100)
        mu, sigma = gp.predict(hs.reshape(-1, 1), return_std=True)
        best = np.min(Ys)
        Z = (best - mu) / np.maximum(sigma, 1e-8)
        cdf = 0.5 * (1 + erf(Z / math.sqrt(2)))
        pdf = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
        ei = (best - mu) * cdf + sigma * pdf
        x_next = hs[np.argmax(ei)]
        Ys.append(scorer.score(predict_fn, x_next))
        Xs = np.append(Xs, x_next)
    best_idx = int(np.argmin(Ys))
    return float(Xs[best_idx])


# ----------------------------------------------------------------------------
# High-level interface
# ----------------------------------------------------------------------------


def select_nw_bandwidth(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "gaussian",
    method: str = "analytic",
    folds: int = 5,
    h_bounds: tuple[float, float] = (0.01, 1.0),
    grid_size: int = 30,
    init_bandwidth: float | None = None,
) -> float:
    """Selects the optimal bandwidth for Nadaraya-Watson regression.

    This function provides a unified interface for various bandwidth selection
    methods for Nadaraya-Watson kernel regression. The analytic method uses
    gradients and Hessians of the cross-validation risk for efficient
    optimization.

    Args:
        X: Input values (univariate predictor variable).
        y: Target values (response variable).
        kernel: Kernel function to use for regression ('gaussian' or
            'epanechnikov').
        method: Bandwidth selection method.
        folds: Number of folds for cross-validation (ignored for 'plugin'
            method).
        h_bounds: (min_bandwidth, max_bandwidth) search bounds.
        grid_size: Number of grid points for 'grid' method.
        init_bandwidth: Initial bandwidth for Newton-based methods. If None,
            uses plug-in rule.

    Returns:
        The optimal bandwidth that minimizes cross-validation risk.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    predict_fn = nw_predict
    a, b = h_bounds
    h_grid = np.linspace(a, b, grid_size)

    if method == "grid":
        return grid_search_cv(X, y, kernel, predict_fn, h_grid, folds=folds)
    elif method == "plugin":
        return plug_in_bandwidth(X)
    elif method == "newton_fd":
        h0 = init_bandwidth or plug_in_bandwidth(X)
        return newton_fd(X, y, kernel, predict_fn, h_init=h0, h_min=a, folds=folds)
    elif method == "analytic":
        h0 = init_bandwidth or plug_in_bandwidth(X)
        return analytic_newton(
            X, y, kernel, predict_fn, h_init=h0, h_min=a, folds=folds
        )
    elif method == "golden":
        return golden_section(X, y, kernel, predict_fn, a, b, folds=folds)
    elif method == "bayes":
        return bayes_opt_bandwidth(X, y, kernel, predict_fn, a, b, folds=folds)
    else:
        raise ValueError(f"Unknown method '{method}'.")
