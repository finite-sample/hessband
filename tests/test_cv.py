import numpy as np
import pytest

from hessband.cv import CVScorer


def dummy_predict(Xtr, ytr, Xte, h, kernel):
    """Simple predictor that returns mean of training targets."""
    return np.mean(ytr) * np.ones_like(Xte)


@pytest.mark.parametrize("folds", [2, 10])
def test_cvscorer_valid_folds(folds):
    X = np.arange(10)
    y = np.arange(10)
    scorer = CVScorer(X, y, folds=folds)
    mse = scorer.score(dummy_predict, h=0.1)
    assert np.isfinite(mse)


def test_cvscorer_invalid_folds():
    X = np.arange(5)
    y = np.arange(5)
    for bad_folds in [0, 1, 6]:
        with pytest.raises(ValueError):
            CVScorer(X, y, folds=bad_folds)
