"""
Hessband: Analytic-Hessian bandwidth selection for univariate kernel smoothers.

This package provides tools for selecting bandwidths for Nadaraya–Watson
regression using analytic derivatives of the leave-one-out cross-validation risk.
The main entry point is `select_nw_bandwidth`, which returns an optimal
bandwidth according to different optimisation strategies, including the
analytic-Hessian method.

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
