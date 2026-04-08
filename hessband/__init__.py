"""
Hessband: Analytic-Hessian bandwidth selection for univariate kernel smoothers.

.. deprecated::
    This package is deprecated. Use `hbw` instead:
    pip install hbw

    See https://github.com/finite-sample/hbw for the maintained package.
"""

import warnings

warnings.warn(
    "hessband is deprecated. Use 'pip install hbw' instead. "
    "See https://github.com/finite-sample/hbw",
    DeprecationWarning,
    stacklevel=2,
)

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
