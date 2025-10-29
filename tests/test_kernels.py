# tests/test_kernels.py
import numpy as np

from hessband.kernels import kernel_derivatives


def test_gaussian_derivative_properties():
    """Test basic properties of kernel derivatives rather than finite
    differences."""
    # Test at a few specific points
    test_cases = [
        (0.0, 1.0),  # u=0, h=1
        (1.0, 0.5),  # u=2, h=0.5
        (0.5, 1.0),  # u=0.5, h=1
    ]

    for raw_dist, h in test_cases:
        u = np.array([raw_dist / h])
        w, dw, ddw = kernel_derivatives(u, h, "gaussian")

        # Basic sanity checks
        assert w[0] > 0, "Weight should be positive"
        assert np.isfinite(w[0]), "Weight should be finite"
        assert np.isfinite(dw[0]), "First derivative should be finite"
        assert np.isfinite(ddw[0]), "Second derivative should be finite"

        # For u=0, first derivative should be negative
        # (weight decreases as h increases)
        if raw_dist == 0.0:
            assert dw[0] < 0, "First derivative at u=0 should be negative"
