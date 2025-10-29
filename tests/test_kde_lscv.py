# tests/test_kde_lscv.py
import numpy as np
from hessband.kde import lscv_generic


def test_lscv_grad_hess_match_fd():
    rng = np.random.RandomState(0)
    x = rng.randn(40)
    for kernel in ["gauss", "epan"]:
        for h in [0.3, 0.5, 1.0]:
            f, g, H = lscv_generic(x, h, kernel)
            eps = 1e-5
            f_plus, _, _ = lscv_generic(x, h + eps, kernel)
            f_minus, _, _ = lscv_generic(x, h - eps, kernel)
            g_num = (f_plus - f_minus) / (2 * eps)
            H_num = (f_plus - 2 * f + f_minus) / (eps**2)
            assert np.allclose(g, g_num, rtol=1e-6, atol=1e-6)
            assert np.allclose(H, H_num, rtol=1e-5, atol=1e-5)
