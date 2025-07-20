import numpy as np
import pytest
from hessband import select_kde_bandwidth, lscv_generic


def test_lscv_generic_returns_finite():
    # Generate random normal data
    x = np.random.randn(30)
    h = 0.2
    score, grad, hess = lscv_generic(x, h, 'gauss')
    assert np.isfinite(score), "Score should be finite"
    assert np.isfinite(grad), "Gradient should be finite"
    assert np.isfinite(hess), "Hessian should be finite"


def test_select_kde_bandwidth_positive():
    x = np.random.randn(50)
    h = select_kde_bandwidth(x, kernel='gauss', method='analytic')
    assert h > 0.0
    assert 0.01 <= h <= 1.0


def test_analytic_vs_grid_close():
    # Small sample to compare analytic and grid search
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    h_analytic = select_kde_bandwidth(x, kernel='gauss', method='analytic', h_bounds=(0.01, 0.5))
    h_grid = select_kde_bandwidth(x, kernel='gauss', method='grid', h_bounds=(0.01, 0.5), grid_size=40)
    # The analytic estimate should be close to the grid optimum
    assert abs(h_analytic - h_grid) < 0.05
