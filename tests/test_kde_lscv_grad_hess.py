# tests/test_kde_lscv_grad_hess.py
import numpy as np
from hessband.kde import lscv_generic


def test_lscv_grad_hess_both_kernels():
    rng = np.random.RandomState(0)
    x = rng.randn(50)
    for kernel in ["gauss", "epan"]:
        for h in [0.3, 0.6, 1.0]:
            f, g, H = lscv_generic(x, h, kernel)
            eps = 1e-5
            f_plus, _, _ = lscv_generic(x, h + eps, kernel)
            f_minus, _, _ = lscv_generic(x, h - eps, kernel)
            g_fd = (f_plus - f_minus) / (2 * eps)
            H_fd = (f_plus - 2 * f + f_minus) / (eps**2)
            assert np.allclose(g, g_fd, rtol=1e-6, atol=1e-6)
            assert np.allclose(H, H_fd, rtol=2e-5, atol=2e-6)
