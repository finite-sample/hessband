"""Regression tests for defects found by the correctness audit."""

from __future__ import annotations

import numpy as np
import pytest

from hessband.cv import CVScorer
from hessband.kde import lscv_generic
from hessband.kernels import kernel_derivatives
from hessband.selectors import analytic_newton, nw_predict, plug_in_bandwidth


def _regression_data(n, seed=1):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.normal(0, 1, n))
    y = np.sin(2 * X) + 0.2 * rng.normal(size=n)
    return X, y


class TestAnalyticNewtonRejectsUnsupportedPredictFn:
    """`analytic_newton` differentiates the Nadaraya-Watson risk in closed
    form. It used to accept and silently discard the documented `predict_fn`,
    so passing any other estimator returned a Nadaraya-Watson bandwidth with
    no indication that the argument had been ignored.
    """

    def test_nw_predict_is_accepted(self):
        X, y = _regression_data(80)
        h = analytic_newton(X, y, "gaussian", nw_predict, h_init=0.4)
        assert h > 0

    def test_other_predict_fn_is_rejected(self):
        X, y = _regression_data(80)

        def other_predict(Xtr, ytr, Xte, h, kernel):
            return np.zeros(len(Xte))

        with pytest.raises(ValueError, match="Nadaraya-Watson"):
            analytic_newton(X, y, "gaussian", other_predict, h_init=0.4)


class TestObjectiveDerivativeConsistency:
    """The reported gradient and Hessian must be the derivatives of the
    reported objective, not of an unnormalised version of it.
    """

    @staticmethod
    def _obj_grad_hess(X, y, h, folds=5, kernel="gaussian"):
        """Re-derive analytic_newton's internal objective from the outside.

        Mirrors the closure in analytic_newton, which is not importable.
        """
        scorer = CVScorer(X, y, folds=folds, kernel=kernel)
        grad = hess = obj = 0.0
        total = 0
        for tr, te in scorer.kf.split(scorer.X):
            Xtr, Xte = scorer.X[tr], scorer.X[te]
            ytr, yte = scorer.y[tr], scorer.y[te]
            u = (Xte[:, None] - Xtr[None, :]) / h
            w, d_w, dd_w = kernel_derivatives(u, h, kernel)
            w_sum = w.sum(axis=1)
            num = (w * ytr).sum(axis=1)
            m = num / w_sum
            residual = yte - m
            obj += np.sum(residual**2)
            d_num = (d_w * ytr).sum(1)
            dd_num = (dd_w * ytr).sum(1)
            d_den = d_w.sum(1)
            dd_den = dd_w.sum(1)
            dm = (d_num * w_sum - num * d_den) / (w_sum**2)
            ddm = (
                dd_num * w_sum
                - 2 * d_num * d_den
                - num * dd_den
                + 2 * num * (d_den**2) / w_sum
            ) / (w_sum**2)
            grad += -2 * np.sum(residual * dm)
            hess += 2 * np.sum(dm**2 - residual * ddm)
            total += len(yte)
        return obj / total, grad / total, hess / total

    def test_analytic_newton_line_search_survives_large_n(self):
        """The Armijo threshold must not scale with the sample size.

        With sums for the derivatives and a mean for the objective, the
        sufficient-decrease requirement is inflated by a factor of n, so for
        n >= 1/c1 = 1e4 the line search can never succeed and the method
        returns its own starting point.
        """
        n, folds = 12000, 20
        X, y = _regression_data(n)
        h0 = plug_in_bandwidth(X)
        h = analytic_newton(
            X, y, "gaussian", nw_predict, h_init=h0, h_min=0.01, folds=folds
        )
        assert abs(h - h0) > 1e-3, (
            f"analytic_newton returned its starting bandwidth ({h:.6f} vs "
            f"h_init={h0:.6f}); the line search never accepted a step"
        )

        scorer = CVScorer(X, y, folds=folds, kernel="gaussian")
        assert scorer.score(nw_predict, h) < scorer.score(nw_predict, h0)


@pytest.mark.parametrize("kernel", ["gauss", "epan"])
def test_lscv_derivatives_match_finite_differences(kernel):
    """Guardrail: the KDE analytic gradient and Hessian are correct.

    Both kernels are smooth in h away from the measure-zero set where some
    |u_ij| hits a support boundary, so central differences converge.
    """
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(0, 1, 120))
    h = 0.35
    s0, g, H = lscv_generic(x, h, kernel)
    eps = 1e-4
    sp = lscv_generic(x, h + eps, kernel)[0]
    sm = lscv_generic(x, h - eps, kernel)[0]
    assert (sp - sm) / (2 * eps) == pytest.approx(g, rel=1e-3)
    assert (sp + sm - 2 * s0) / eps**2 == pytest.approx(H, rel=1e-3)


def test_kernel_derivatives_match_finite_differences():
    """Guardrail: d/dh and d2/dh2 of K(u)/h for the Gaussian kernel."""
    delta = np.array([-0.7, 0.0, 0.31, 1.4])
    h, eps = 0.4, 1e-4
    w, dw, d2w = kernel_derivatives(delta / h, h, "gaussian")
    wp = kernel_derivatives(delta / (h + eps), h + eps, "gaussian")[0]
    wm = kernel_derivatives(delta / (h - eps), h - eps, "gaussian")[0]
    np.testing.assert_allclose((wp - wm) / (2 * eps), dw, rtol=1e-4)
    np.testing.assert_allclose((wp + wm - 2 * w) / eps**2, d2w, rtol=1e-4)
