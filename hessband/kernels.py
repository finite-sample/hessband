# kernels.py
# Univariate kernel weights and derivatives with respect to bandwidth h.
# Conventions:
#   u = (x - x') / h
#   w(h) = K(u) / h
# Derivatives (ignoring compact-support boundary terms):
#   dw/dh  = -(K + u K') / h^2
#   d2w/dh2 = (2K + 4u K' + u^2 K'') / h^3
#
# For the Gaussian kernel this yields:
#   dw/dh  = w * (u^2 - 1) / h
#   d2w/dh2 = w * (u^4 - 5u^2 + 2) / h^2

from __future__ import annotations

import numpy as np

__all__ = [
    "weights_gaussian",
    "weights_epanechnikov",
    "kernel_derivatives",
    "kernel_weights",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def weights_gaussian(u: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel weight K(u)/h with K(u)=phi(u)."""
    u = np.asarray(u, dtype=float)
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive")
    K = np.exp(-0.5 * u * u) / _SQRT_2PI
    return K / h


def weights_epanechnikov(u: np.ndarray, h: float) -> np.ndarray:
    """Epanechnikov kernel weight K(u)/h with K(u)=0.75*(1-u^2) on |u|<=1."""
    u = np.asarray(u, dtype=float)
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive")
    base = 0.75 * np.maximum(0.0, 1.0 - u * u)
    return base / h


def kernel_derivatives(u: np.ndarray, h: float, kernel: str):
    """
    Return (w, dw, d2w) for w(h)=K(u)/h where u=(x-x')/h.
    Derivatives are with respect to h and ignore support-boundary motion
    for compact kernels, which is the standard practical convention.

    Parameters
    ----------
    u : array-like
        Normalized pairwise distances (x - x') / h.
    h : float
        Bandwidth (> 0).
    kernel : {"gaussian", "epanechnikov", "epan"}
        Kernel name.

    Returns
    -------
    w : ndarray
        K(u) / h.
    dw : ndarray
        d/dh [ K(u)/h ].
    d2w : ndarray
        d^2/dh^2 [ K(u)/h ].
    """
    u = np.asarray(u, dtype=float)
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive")

    k = kernel.lower()
    if k == "gaussian":
        K = np.exp(-0.5 * u * u) / _SQRT_2PI
        Kp = -u * K
        Kpp = (u * u - 1.0) * K
    elif k in ("epanechnikov", "epan"):
        mask = np.abs(u) <= 1.0
        K = np.zeros_like(u, dtype=float)
        Kp = np.zeros_like(u, dtype=float)
        Kpp = np.zeros_like(u, dtype=float)
        um = u[mask]
        K[mask] = 0.75 * (1.0 - um * um)
        Kp[mask] = -1.5 * um
        Kpp[mask] = -1.5
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    w = K / h
    dw = -(K + u * Kp) / (h * h)
    d2w = (2.0 * K + 4.0 * u * Kp + (u * u) * Kpp) / (h**3)
    return w, dw, d2w


def kernel_weights(u: np.ndarray, h: float, kernel: str) -> np.ndarray:
    """
    Compute kernel weights K(u)/h for given kernel.

    Parameters
    ----------
    u : array-like
        Normalized pairwise distances (x - x') / h.
    h : float
        Bandwidth (> 0).
    kernel : {"gaussian", "epanechnikov", "epan"}
        Kernel name.

    Returns
    -------
    ndarray
        K(u) / h.
    """
    k = kernel.lower()
    if k == "gaussian":
        return weights_gaussian(u, h)
    elif k in ("epanechnikov", "epan"):
        return weights_epanechnikov(u, h)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
