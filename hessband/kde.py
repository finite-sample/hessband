"""
Kernel density estimation (KDE) bandwidth selectors with analytic gradients.

This module implements leave‑one‑out least‑squares cross‑validation (LSCV) for
univariate KDE with Gaussian and Epanechnikov kernels.  It provides analytic
expressions for the cross‑validation score, gradient and Hessian with respect
to the bandwidth.  A Newton–Armijo optimiser is included to select the optimal
bandwidth without numerical differencing.

The analytic formulas are based on convolution of kernels and their
derivatives; see the accompanying paper for details.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Kernel primitives & helper functions
# ---------------------------------------------------------------------------
SQRT_2PI = np.sqrt(2 * np.pi)


def _pairwise_sq(x):
    """Compute n×n squared distances matrix."""
    return (x[:, None] - x[None, :]) ** 2


def _poly_mask(u: np.ndarray, mask: np.ndarray, expr):
    """Return expr on mask, 0 elsewhere. expr may be scalar or array."""
    out = np.zeros_like(u, dtype=float)
    if np.isscalar(expr):
        out[mask] = expr
    else:
        out[mask] = expr[mask]
    return out


def _pairwise_sums(
    u: np.ndarray,
    K: Callable[[np.ndarray], np.ndarray],
    Kp: Callable[[np.ndarray], np.ndarray],
    Kpp: Callable[[np.ndarray], np.ndarray],
    K2: Callable[[np.ndarray], np.ndarray],
    K2p: Callable[[np.ndarray], np.ndarray],
    K2pp: Callable[[np.ndarray], np.ndarray],
    off: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute the paired sums that appear in the LSCV gradient and Hessian.

    S_F  = sum_ij [ K2(u) + u K2'(u) ]
    S_Fu = sum_ij [ u ( 2 K2'(u) + u K2''(u) ) ]
    S_K  = sum_{i != j} [ K(u) + u K'(u) ]
    S_Ku = sum_{i != j} [ u ( 2 K'(u) + u K''(u) ) ]
    """
    K2u = K2(u)
    K2pu = K2p(u)
    K2ppu = K2pp(u)

    S_F = (K2u + u * K2pu).sum()
    S_Fu = (u * (2.0 * K2pu + u * K2ppu)).sum()

    Ku = K(u)
    Kpu = Kp(u)
    Kppu = Kpp(u)

    S_K = (Ku + u * Kpu)[off].sum()
    S_Ku = (u * (2.0 * Kpu + u * Kppu))[off].sum()
    return float(S_F), float(S_Fu), float(S_K), float(S_Ku)


# Gaussian kernel and derivatives


def K_gauss(u):
    """Gaussian kernel."""
    return np.exp(-0.5 * u * u) / SQRT_2PI


def K_gauss_p(u):
    """First derivative of Gaussian kernel."""
    return -u * K_gauss(u)


def K_gauss_pp(u):
    """Second derivative of Gaussian kernel."""
    return (u * u - 1) * K_gauss(u)


def K2_gauss(u):
    """Convolution of Gaussian kernel with itself."""
    return np.exp(-0.25 * u * u) / np.sqrt(4 * np.pi)


def K2_gauss_p(u):
    """First derivative of Gaussian convolution."""
    return -0.5 * u * K2_gauss(u)


def K2_gauss_pp(u):
    """Second derivative of Gaussian convolution."""
    return (0.25 * u * u - 0.5) * K2_gauss(u)


# Epanechnikov kernel and derivatives (piecewise)
def _absu(u):
    """Absolute value helper."""
    return np.abs(u)


def K_epan(u):
    """Epanechnikov kernel."""
    return _poly_mask(u, _absu(u) <= 1, 0.75 * (1 - u * u))


def K_epan_p(u):
    """First derivative of Epanechnikov kernel."""
    return _poly_mask(u, _absu(u) <= 1, -1.5 * u)


def K_epan_pp(u):
    """Second derivative of Epanechnikov kernel."""
    return _poly_mask(u, _absu(u) <= 1, -1.5)


# Convolution K*K for Epanechnikov kernel (valid for |u| ≤ 2)
def K2_epan(u):
    """Convolution of Epanechnikov kernel with itself."""
    return _poly_mask(
        u,
        _absu(u) <= 2,
        (0.6 - 0.75 * _absu(u) ** 2 + 0.375 * _absu(u) ** 3 - 0.01875 * _absu(u) ** 5),
    )


def K2_epan_p(u):
    """First derivative of Epanechnikov convolution."""
    return _poly_mask(
        u,
        _absu(u) <= 2,
        np.sign(u)
        * (-0.09375 * _absu(u) ** 4 + 1.125 * _absu(u) ** 2 - 1.5 * _absu(u)),
    )


def K2_epan_pp(u):
    """Second derivative of Epanechnikov convolution."""
    return _poly_mask(
        u,
        _absu(u) <= 2,
        -0.375 * _absu(u) ** 3 + 2.25 * _absu(u) - 1.5,
    )


KERNELS: Dict[
    str, Tuple[Callable, Callable, Callable, Callable, Callable, Callable]
] = {
    "gauss": (
        K_gauss,
        K_gauss_p,
        K_gauss_pp,
        K2_gauss,
        K2_gauss_p,
        K2_gauss_pp,
    ),
    "epan": (K_epan, K_epan_p, K_epan_pp, K2_epan, K2_epan_p, K2_epan_pp),
}

# ---------------------------------------------------------------------------
# LSCV score, gradient and Hessian
# ---------------------------------------------------------------------------


def lscv_generic(x: np.ndarray, h: float, kernel: str):
    """
    Least-squares cross-validation for univariate KDE with analytic
    gradient and Hessian with respect to h.

    LSCV(h) = 1/(n^2 h) sum_{i,j} K2(u_ij)
             - 2/(n(n-1) h) sum_{i != j} K(u_ij),
    where u_ij = (x_i - x_j)/h and K2 is the kernel convolution K * K.

    Returns
    -------
    score : float
    grad  : float
    hess  : float
    """
    if h <= 0:
        raise ValueError("h must be positive")
    K, Kp, Kpp, K2, K2p, K2pp = KERNELS[kernel]
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")

    u = (x[:, None] - x[None, :]) / h
    off = ~np.eye(n, dtype=bool)

    # score
    term1 = K2(u).sum() / (n**2 * h)
    term2 = K(u)[off].sum() / (n * (n - 1) * h)
    score = float(term1 - 2.0 * term2)

    # gradient and Hessian
    S_F, S_Fu, S_K, S_Ku = _pairwise_sums(u, K, Kp, Kpp, K2, K2p, K2pp, off)

    grad = -S_F / (n**2 * h**2) + 2.0 * S_K / (n * (n - 1) * h**2)

    hess = (2.0 * S_F + S_Fu) / (n**2 * h**3) - (2.0 * S_Ku + 4.0 * S_K) / (
        n * (n - 1) * h**3
    )

    return score, float(grad), float(hess)


def lscv_gauss(x, h):
    """LSCV for Gaussian kernel."""
    return lscv_generic(x, h, "gauss")


def lscv_epan(x, h):
    """LSCV for Epanechnikov kernel."""
    return lscv_generic(x, h, "epan")


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


def select_kde_bandwidth(
    x: np.ndarray,
    kernel: str = "gauss",
    method: str = "analytic",
    h_bounds=(0.01, 1.0),
    grid_size: int = 30,
    h_init: Optional[float] = None,
) -> float:
    """
    Select an optimal bandwidth for univariate kernel density estimation using LSCV.

    This function minimizes the least-squares cross-validation (LSCV) criterion
    to select an optimal bandwidth for kernel density estimation. The analytic
    method uses exact gradients and Hessians for efficient Newton optimization.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Data samples for density estimation.
    kernel : {'gauss', 'epan'}, default='gauss'
        Kernel function:

        - 'gauss': Gaussian (normal) kernel
        - 'epan': Epanechnikov kernel (compact support)
    method : {'analytic', 'grid', 'golden'}, default='analytic'
        Bandwidth selection method:

        - 'analytic': Newton–Armijo with analytic derivatives (recommended)
        - 'grid': Exhaustive grid search over h_bounds
        - 'golden': Golden-section search optimization
    h_bounds : tuple of float, default=(0.01, 1.0)
        (min_bandwidth, max_bandwidth) search bounds.
    grid_size : int, default=30
        Number of grid points for 'grid' method.
    h_init : float, optional
        Initial bandwidth for Newton-based methods. If None, uses Silverman's
        rule of thumb as starting point.

    Returns
    -------
    float
        Optimal bandwidth that minimizes LSCV criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from hessband import select_kde_bandwidth
    >>> # Generate sample data from mixture distribution
    >>> x = np.concatenate([
    ...     np.random.normal(-2, 0.5, 200),
    ...     np.random.normal(2, 1.0, 300)
    ... ])
    >>> # Select bandwidth using analytic method
    >>> h_opt = select_kde_bandwidth(x, kernel='gauss', method='analytic')
    >>> print(f"Optimal bandwidth: {h_opt:.4f}")

    Notes
    -----
    The LSCV criterion is defined as:

    LSCV(h) = ∫ f̂ₕ²(x) dx - 2∫ f̂ₕ(x) f(x) dx

    where f̂ₕ is the kernel density estimate with bandwidth h and f is the true
    (unknown) density. The analytic method provides exact derivatives, making
    optimization very efficient compared to finite-difference approaches.
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
        h_opt, _ = newton_opt(
            x, h_init, lambda x_arr, h_val: lscv_generic(x_arr, h_val, kernel)
        )
        return float(h_opt)
    else:
        raise ValueError(f"Unknown method '{method}'.")


__all__ = [
    "select_kde_bandwidth",
    "lscv_generic",
    "lscv_gauss",
    "lscv_epan",
]
