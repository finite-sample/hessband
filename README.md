# Hessband: Analytic Bandwidth Selector

[![image](https://github.com/finite-sample/hessband/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/hessband/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/hessband.svg)](https://pypi.org/project/hessband/)
[![PyPI Downloads](https://static.pepy.tech/badge/hessband)](https://pepy.tech/projects/hessband)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/hessband/)

Hessband is a Python package for selecting bandwidths in univariate smoothing.  It provides analytic gradients and Hessians of the leave‑one‑out cross‑validation (LOOCV) risk for Nadaraya–Watson regression and least‑squares cross‑validation (LSCV) for kernel density estimation (KDE).  Bandwidth selectors include grid search, plug‑in rules, finite‑difference Newton, analytic Newton, golden‑section search, and Bayesian optimisation.

## Installation

To install from source, navigate to the directory containing `hessband` and run:

```bash
pip install hessband-0.1.0.tar.gz
```

Alternatively, you can unpack the tarball and install using `setup.py`:

```bash
tar -xzvf hessband-0.1.0.tar.gz
cd hessband
pip install .
```

## Usage Example

```python
import numpy as np
from hessband import select_nw_bandwidth, nw_predict

# Generate synthetic data
X = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(200)

# Select the optimal bandwidth via the analytic-Hessian method
h_opt = select_nw_bandwidth(X, y, method='analytic', kernel='gaussian')

# Predict on the original points
y_pred = nw_predict(X, y, X, h_opt)

print("Selected bandwidth:", h_opt)
print("Mean squared error:", np.mean((y_pred - np.sin(2 * np.pi * X)) ** 2))
```

When running the example above, you should see a selected bandwidth around `0.16` and a mean squared error close to `8e-4`. Results may vary slightly due to randomness in the synthetic data.

### KDE Example

The package also supports bandwidth selection for univariate kernel density estimation using least‑squares cross‑validation (LSCV).  For example:

```python
import numpy as np
from hessband import select_kde_bandwidth

# Sample data from a bimodal distribution
x = np.concatenate([
    np.random.normal(-2, 0.5, 200),
    np.random.normal(2, 1.0, 200),
])

# Select bandwidth using analytic Newton for the Gaussian kernel
h_kde = select_kde_bandwidth(x, kernel='gauss', method='analytic')
print("Selected KDE bandwidth:", h_kde)
```

The `select_kde_bandwidth` function also supports Epanechnikov kernels (`kernel='epan'`), grid search (`method='grid'`) and golden‑section optimisation (`method='golden'`).

## Simulation Results

In the accompanying paper, we compared several bandwidth selectors using simulated data from a bimodal mixture regression model. A subset of the results for the Gaussian kernel with noise level `0.1` and sample size `200` is given below:

| Method               | MSE (×10⁻³)       | CV evaluations |
|----------------------|-------------------|---------------|
| Grid                 | 0.87 ± 0.12       | 150 ± 0       |
| Plug-in              | 6.31 ± 0.57       | 5 ± 0         |
| Finite-diff Newton   | 6.31 ± 0.57       | 20 ± 0        |
| **Analytic Newton**  | **0.86 ± 0.13**   | **0 ± 0**     |
| Golden               | 0.86 ± 0.13       | 85 ± 0        |
| Bayes                | 0.87 ± 0.14       | 75 ± 0        |

The analytic-Hessian method matches the accuracy of exhaustive grid search while requiring essentially no cross-validation evaluations.
