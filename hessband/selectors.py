"""
Bandwidth selection methods for univariate kernel regression.

This module provides several bandwidth selectors, including grid search,
plug-in rules, finite-difference Newton, analytic Newton (analytic-Hessian),
golden-section search and Bayesian optimisation. A high-level function
`select_nw_bandwidth` orchestrates the selection process.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
import math
from math import erf
from .kernels import kernel_weights, kernel_derivatives
from .cv import CVScorer

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

def nw_predict(X_train, y_train, X_test, h, kernel='gaussian'):
    """Compute Nadaraya–Watson predictions using a specified kernel."""
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
    return numerator / denom

# ----------------------------------------------------------------------------
# Basic bandwidth selectors
# ----------------------------------------------------------------------------

def grid_search_cv(X, y, kernel, predict_fn, h_grid, folds=5):
    """Grid search for the best bandwidth using cross-validation."""
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    best_h, best_score = None, np.inf
    for h in h_grid:
        score = scorer.score(predict_fn, h)
        if score < best_score:
            best_score, best_h = score, h
    return best_h

def plug_in_bandwidth(X):
    """Plug-in bandwidth based on Silverman's rule of thumb."""
    sigma = np.std(np.asarray(X).ravel(), ddof=1)
    n = len(X)
    return 1.06 * sigma * n ** (-1/5)

def newton_fd(X, y, kernel, predict_fn, h_init, h_min=1e-3, folds=5,
              tol=1e-3, max_iter=10, eps=1e-4):
    """Finite-difference Newton method for bandwidth selection."""
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)
    h = max(h_init, h_min)
    for _ in range(max_iter):
        s0 = scorer.score(predict_fn, h)
        s1 = scorer.score(predict_fn, h + eps)
        s_1 = scorer.score(predict_fn, max(h - eps, h_min))
        grad = (s1 - s_1) / (2 * eps)
        hess = (s1 + s_1 - 2 * s0) / (eps ** 2)
        if hess <= 0:
            break
        h_new = max(h_min, h - grad / hess)
        if abs(h_new - h) < tol:
            h = h_new
            break
        h = h_new
    return h

def analytic_newton(X, y, kernel, predict_fn, h_init, h_min=1e-3,
                    folds=5, tol=1e-3, max_iter=10):
    """
    Analytic Newton method for LOOCV risk minimisation.
    Returns the bandwidth without performing CV evaluations in the loop.
    """
    scorer = CVScorer(X, y, folds=folds, kernel=kernel)

    def obj_grad_hess(h):
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
                dd_num * w_sum - 2 * d_num * d_den - num * dd_den
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

def golden_section(X, y, kernel, predict_fn, a, b, folds=5,
                   tol=1e-3, max_iter=20):
    """Golden-section search for bandwidth selection."""
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

def bayes_opt_bandwidth(X, y, kernel, predict_fn, a, b,
                        folds=5, init_points=5, n_iter=10):
    """Bayesian optimisation for bandwidth selection."""
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
                gp = GaussianProcessRegressor(kernel=kernel_gp, normalize_y=True).fit(X_train, y_train)
                # If noise_level hit lower bound, relax it
                if any(issubclass(wi.category, warnings.ConvergenceWarning) for wi in w):
                    lb, ub = wk.noise_level_bounds
                    new_lb = max(lb / 10, 1e-8)
                    wk = WhiteKernel(noise_level=wk.noise_level, noise_level_bounds=(new_lb, ub))
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

def select_nw_bandwidth(X, y, kernel='gaussian', method='analytic',
                        folds=5, h_bounds=(0.01, 1.0), grid_size=30,
                        init_bandwidth=None):
    """
    Select the optimal bandwidth for Nadaraya–Watson regression.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Input values.
    y : array-like, shape (n_samples,)
        Target values.
    kernel : str, optional (default='gaussian')
        Kernel type ('gaussian' or 'epanechnikov').
    method : str, optional (default='analytic')
        Bandwidth selection method: one of {'analytic', 'grid', 'plugin',
        'newton_fd', 'golden', 'bayes'}.
    folds : int, optional (default=5)
        Number of folds for cross-validation.
    h_bounds : tuple, optional (default=(0.01, 1.0))
        Lower and upper bounds for the bandwidth search.
    grid_size : int, optional (default=30)
        Number of grid points for grid search.
    init_bandwidth : float, optional
        Initial bandwidth for Newton-based methods. If None, uses plug-in rule.

    Returns
    -------
    float
        Selected bandwidth.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    predict_fn = nw_predict
    a, b = h_bounds
    h_grid = np.linspace(a, b, grid_size)

    if method == 'grid':
        return grid_search_cv(X, y, kernel, predict_fn, h_grid, folds=folds)
    elif method == 'plugin':
        return plug_in_bandwidth(X)
    elif method == 'newton_fd':
        h0 = init_bandwidth or plug_in_bandwidth(X)
        return newton_fd(X, y, kernel, predict_fn, h_init=h0, h_min=a,
                         folds=folds)
    elif method == 'analytic':
        h0 = init_bandwidth or plug_in_bandwidth(X)
        return analytic_newton(X, y, kernel, predict_fn, h_init=h0,
                               h_min=a, folds=folds)
    elif method == 'golden':
        return golden_section(X, y, kernel, predict_fn, a, b,
                              folds=folds)
    elif method == 'bayes':
        return bayes_opt_bandwidth(X, y, kernel, predict_fn, a, b,
                                   folds=folds)
    else:
        raise ValueError(f"Unknown method '{method}'.")
