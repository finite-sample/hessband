"""
Hessband: Analytic-Hessian bandwidth selection for univariate kernel smoothers.

This package provides tools for selecting bandwidths for Nadaraya–Watson
regression and kernel density estimation (KDE) using analytic derivatives
of cross-validation risk functions. It supports both leave-one-out
cross-validation (LOOCV) for regression and least-squares cross-validation
(LSCV) for density estimation.

Key Features
------------
- Analytic gradients and Hessians for efficient optimization
- Multiple bandwidth selection methods (Newton, grid search, golden section, Bayesian)
- Support for Gaussian and Epanechnikov kernels
- Fast implementations with minimal cross-validation evaluations

Main Functions
--------------
select_nw_bandwidth : Select optimal bandwidth for Nadaraya-Watson regression
select_kde_bandwidth : Select optimal bandwidth for kernel density estimation
nw_predict : Make predictions using Nadaraya-Watson estimator
lscv_generic : Compute LSCV score with analytic derivatives

Example
-------
>>> import numpy as np
>>> from hessband import select_nw_bandwidth, nw_predict
>>> # Generate synthetic data
>>> X = np.linspace(0, 1, 200)
>>> y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(200)
>>> # Select bandwidth via analytic-Hessian method
>>> h_opt = select_nw_bandwidth(X, y, method='analytic')
>>> # Predict at new points
>>> y_pred = nw_predict(X, y, X, h_opt)

For KDE example:
>>> from hessband import select_kde_bandwidth
>>> x = np.random.normal(0, 1, 1000)
>>> h_kde = select_kde_bandwidth(x, kernel='gauss', method='analytic')
"""

from .kde import lscv_generic, select_kde_bandwidth
from .selectors import (
    analytic_newton,
    bayes_opt_bandwidth,
    golden_section,
    grid_search_cv,
    newton_fd,
    nw_predict,
    plug_in_bandwidth,
    select_nw_bandwidth,
)

__all__ = [
    "select_nw_bandwidth",
    "nw_predict",
    "grid_search_cv",
    "plug_in_bandwidth",
    "newton_fd",
    "analytic_newton",
    "golden_section",
    "bayes_opt_bandwidth",
    "select_kde_bandwidth",
    "lscv_generic",
]
