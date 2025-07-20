"""
Kernel density estimation (KDE) bandwidth selectors with analytic gradients and Hessians.

This module implements leave‑one‑out least‑squares cross‑validation (LSCV) for
univariate KDE with Gaussian and Epanechnikov kernels.  It provides analytic
expressions for the cross‑validation score, gradient and Hessian with respect to
the bandwidth.  A Newton–Armijo optimiser is included to select the optimal
bandwidth without numerical differencing.

The analytic formulas are based on convolution of kernels and their
derivatives; see the accompanying paper for details.
"""

from typing import Callable, Tuple, Dict
import numpy as np

# ---------------------------------------------------------------------------
# Kernel primitives & helper functions
# ---------------------------------------------------------------------------
SQRT_2PI = np.sqrt(2 * np.pi)
_pairwise_sq = lambda x: (x[:, None] - x[None, :]) ** 2  # n×n squared distances

def _poly_mask(u: np.ndarray, mask: np.ndarray, expr):
    """Return expr on mask, 0 elsewhere. expr may be scalar or array."""
    out = np.zeros_like(u, dtype=float)
    if np.isscalar(expr):
        out[mask] = expr
    else:
        out[mask] = expr[mask]
    return out

# Gaussian kernel and derivatives
K_gauss = lambda u: np.exp(-0.5 * u * u) / SQRT_2PI
K_gauss_p = lambda u: -u * K_gauss(u)
K_gauss_pp = lambda u: (u * u - 1) * K_gauss(u)
K2_gauss = lambda u: np.exp(-0.25 * u * u) / np.sqrt(4 * np.pi)
K2_gauss_p = lambda u: -0.5 * u * K2_gauss(u)
K2_gauss_pp = lambda u: (0.25 * u * u - 0.5) * K2_gauss(u)

# Epanechnikov kernel and derivatives (piecewise)
_absu = lambda u: np.abs(u)
K_epan = lambda u: _poly_mask(u, _absu(u) <= 1, 0.75 * (1 - u * u))
K_epan_p = lambda u: _poly_mask(u, _absu(u) <= 1, -1.5 * u)
K_epan_pp = lambda u: _poly_mask(u, _absu(u) <= 1, -1.5)
# Convolution K*K for Epanechnikov kernel (valid for |u| ≤ 2)
K2_epan = lambda u: _poly_mask(
    u,
    _absu(u) <= 2,
    0.6 - 0.75 * _absu(u) ** 2 + 0.375 * _absu(u) ** 3 - 0.01875 * _absu(u) ** 5,
)
K2_epan_p = lambda u: _poly_mask(
    u,
    _absu(u) <= 2,
    np.sign(u) * (-0.09375 * _absu(u) ** 4 + 1.125 * _absu(u) ** 2 - 1.5 * _absu(u)),
)
K2_epan_pp = lambda u: _poly_mask(
    u,
    _absu(u) <= 2,
    -0.375 * _absu(u) ** 3 + 2.25 * _absu(u) - 1.5,
)

KERNELS: Dict[str, Tuple[Callable, Callable, Callable, Callable, Callable, Callable]] = {
    "gauss": (K_gauss, K_gauss_p, K_gauss_pp, K2_gauss, K2_gauss_p, K2_gauss_pp),
    "epan": (K_epan, K_epan_p, K_epan_pp, K2_epan, K2_epan_p, K2_epan_pp),
}

# ---------------------------------------------------------------------------
# LSCV score, gradient and Hessian
# ---------------------------------------------------------------------------

def lscv_generic(x: np.ndarray, h: float, kernel: str) -> Tuple[float, float, float]:
    """Return (LSCV, gradient, Hessian) at bandwidth h for the chosen kernel."""
    K, Kp, Kpp, K2, K2p, K2pp = KERNELS[kernel]
    n = len(x)
    u = (x[:, None] - x[None, :]) / h
    # Score
    term1 = K2(u).sum() / (n ** 2 * h)
    K_vals = K(u)
    term2 = (K_vals.sum() - np.sum(np.diag(K_vals))) / (n * (n - 1) * h)
    score = term1 - 2 * term2
    # Gradient
    S_F = (K2(u) + u * K2p(u)).sum()
    S_K = (K_vals + u * Kp(u)).sum() - 0.0  # diagonal excluded automatically
    grad = -S_F / (n ** 2 * h ** 2) + 2 * S_K / (n * (n - 1) * h ** 2)
    # Hessian
    S_F2 = (2 * K2p(u) + u * K2pp(u)).sum()
    S_K2 = (2 * Kp(u) + u * Kpp(u)).sum()
    hess = 2 * S_F / (n ** 2 * h ** 3) - S_F2 / (n ** 2 * h ** 2)
    hess += -4 * S_K / (n * (n - 1) * h ** 3) + 2 * S_K2 / (n * (n - 1) * h ** 2)
    return score, grad, hess

lscv_gauss = lambda x, h: lscv_generic(x, h, "gauss")
lscv_epan = lambda x, h: lscv_generic(x, h, "epan")

# ---------------------------------------------------------------------------
# Newton–Armijo optimiser for scalar bandwidth
# ---------------------------------------------------------------------------

def newton_opt(
    x: np.ndarray,
    h0: float,
    score_grad_hess: Callable[[np.ndarray, float], Tuple[float, float, float]],
    tol: float = 1e-5,
    max_iter: int = 12,
) -> Tuple[float, int]:
    """Newton–Armijo optimisation to minimise the LSCV score."""
    h, evals = h0, 0
    for _ in range(max_iter):
        f, g, H = score_grad_hess(x, h)
        evals += 1
        if abs(g) < tol:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        # Armijo back‑tracking (cheap, no new Jacobians)
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            new_f, _, _ = score_grad_hess(x, h_new)
            if new_f < f:
                h = h_new
                break
            step *= 0.5
    return h, evals

# ---------------------------------------------------------------------------
# High-level bandwidth selector
# ---------------------------------------------------------------------------

def select_kde_bandwidth(x: np.ndarray, kernel: str = "gauss",
                         method: str = "analytic", h_bounds=(0.01, 1.0),
                         grid_size: int = 30, h_init: float = None) -> float:
    """
    Select an optimal bandwidth for univariate KDE using LSCV.

    Parameters
    ----------
    x : array-like
        Data samples.
    kernel : str, optional
        Kernel name: 'gauss' or 'epan'.
    method : str, optional
        Selection method: 'analytic' (Newton–Armijo), 'grid', or 'golden'.
    h_bounds : tuple, optional
        Lower and upper bounds for the search.
    grid_size : int, optional
        Number of grid points for grid search.
    h_init : float, optional
        Initial bandwidth for Newton optimisation. Defaults to plug-in estimate.

    Returns
    -------
    float
        Selected bandwidth.
    """
    x = np.asarray(x).ravel()
    a, b = h_bounds
    # Plug-in rule for initial bandwidth: Silverman's rule of thumb
    if h_init is None:
        sigma = np.std(x, ddof=1)
        n = len(x)
        h_init = 1.06 * sigma * n ** (-1 / 5)
    if method == "grid":
        h_grid = np.logspace(np.log10(a), np.log10(b), grid_size)
        scores = [lscv_generic(x, h, kernel)[0] for h in h_grid]
        return float(h_grid[int(np.argmin(scores))])
    elif method == "golden":
        # Golden-section search on log-scale
        phi = (1 + np.sqrt(5)) / 2
        log_a, log_b = np.log(a), np.log(b)
        c = log_b - (log_b - log_a) / phi
        d = log_a + (log_b - log_a) / phi
        f_c = lscv_generic(x, np.exp(c), kernel)[0]
        f_d = lscv_generic(x, np.exp(d), kernel)[0]
        for _ in range(20):
            if abs(log_b - log_a) < 1e-3:
                break
            if f_c < f_d:
                log_b, f_d = d, f_c
                d = c
                c = log_b - (log_b - log_a) / phi
                f_c = lscv_generic(x, np.exp(c), kernel)[0]
            else:
                log_a, f_c = c, f_d
                c = d
                d = log_a + (log_b - log_a) / phi
                f_d = lscv_generic(x, np.exp(d), kernel)[0]
        return float(np.exp((log_a + log_b) / 2))
    elif method == "analytic":
        # Use Newton–Armijo on the LSCV objective
        h_opt, _ = newton_opt(x, h_init, lambda x_arr, h_val: lscv_generic(x_arr, h_val, kernel))
        return float(h_opt)
    else:
        raise ValueError(f"Unknown method '{method}'.")

__all__ = [
    "select_kde_bandwidth",
    "lscv_generic",
    "lscv_gauss",
    "lscv_epan",
]
