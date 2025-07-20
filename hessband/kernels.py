"""
Kernel functions and derivatives for univariate smoothing.

This module provides routines to compute kernel weights and their
first and second derivatives with respect to the bandwidth for
Gaussian and Epanechnikov kernels.

These functions are used internally by the analytic-Hessian bandwidth
selector.
"""

import numpy as np

__all__ = [
    "weights_gaussian",
    "weights_epanechnikov",
    "kernel_weights",
    "kernel_derivatives",
]

def weights_gaussian(u: np.ndarray, h: float):
    """Return Gaussian weights for scaled distances u and bandwidth h."""
    return np.exp(-0.5 * u**2) / (h * np.sqrt(2 * np.pi))

def weights_epanechnikov(u: np.ndarray, h: float):
    """Return Epanechnikov weights for scaled distances u and bandwidth h."""
    w = np.zeros_like(u)
    mask = np.abs(u) <= 1
    w[mask] = 0.75 * (1 - u[mask]**2) / h
    return w

def kernel_weights(u: np.ndarray, h: float, kernel: str = "gaussian"):
    """Dispatch to the appropriate kernel weight function."""
    if kernel == "gaussian":
        return weights_gaussian(u, h)
    elif kernel == "epanechnikov":
        return weights_epanechnikov(u, h)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

def kernel_derivatives(u: np.ndarray, h: float, kernel: str):
    """
    Compute the first and second derivatives of kernel weights with respect to h.

    Returns a tuple (w, d_w, dd_w), where w are the weights, d_w the derivative
    and dd_w the second derivative. The derivatives are computed analytically
    for the Gaussian and Epanechnikov kernels.
    """
    if kernel == "gaussian":
        w = weights_gaussian(u, h)
        d_w = w * ((u**2 - 1) / h)
        dd_w = w * ((u**4 - 3 * u**2 + 1) / (h**2))
    elif kernel == "epanechnikov":
        # Epanechnikov kernel with compact support
        mask = np.abs(u) <= 1
        w = np.zeros_like(u)
        d_w = np.zeros_like(u)
        dd_w = np.zeros_like(u)
        w[mask] = 0.75 * (1 - u[mask]**2) / h
        d_w[mask] = 0.75 * ((-1 + 3 * u[mask]**2) / (h**2))
        dd_w[mask] = 1.5 * ((1 - 6 * u[mask]**2) / (h**3))
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")
    return w, d_w, dd_w
