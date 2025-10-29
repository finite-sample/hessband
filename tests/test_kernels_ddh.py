import numpy as np

from hessband.kernels import kernel_derivatives


def test_gaussian_second_derivative_properties():
    """Test basic properties of second derivative of Gaussian kernel."""
    # Test that kernel_derivatives function works for both kernels
    test_points = np.array([0.0, 0.5, 1.0, 2.0])
    h = 1.0

    for kernel in ["gaussian", "epanechnikov"]:
        w, dw, ddw = kernel_derivatives(test_points, h, kernel)

        # Basic checks
        assert len(w) == len(test_points)
        assert len(dw) == len(test_points)
        assert len(ddw) == len(test_points)
        assert np.all(np.isfinite(w))
        assert np.all(np.isfinite(dw))
        assert np.all(np.isfinite(ddw))

        # Weights should be non-negative
        assert np.all(w >= 0)
