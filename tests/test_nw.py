import numpy as np

from hessband import nw_predict, select_nw_bandwidth


def test_nw_predict_constant():
    X = np.linspace(0, 1, 10)
    y = np.ones_like(X) * 5.0
    y_pred = nw_predict(X, y, X, h=0.1, kernel="gaussian")
    assert np.allclose(y_pred, 5.0)


def test_select_nw_bandwidth_positive():
    X = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(len(X))
    h = select_nw_bandwidth(X, y, method="analytic", kernel="gaussian")
    assert h > 0.0
    assert 0.01 <= h <= 1.0


def test_nw_analytic_vs_grid():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, 40)
    y = np.sin(2 * np.pi * X) + 0.05 * rng.standard_normal(40)
    h_analytic = select_nw_bandwidth(
        X, y, method="analytic", kernel="gaussian", h_bounds=(0.01, 0.5)
    )
    h_grid = select_nw_bandwidth(
        X,
        y,
        method="grid",
        kernel="gaussian",
        h_bounds=(0.01, 0.5),
        grid_size=40,
    )
    assert abs(h_analytic - h_grid) < 0.05
