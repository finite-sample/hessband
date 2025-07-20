# Hessband

Hessband is a Python package for selecting bandwidths in univariate Nadaraya–Watson regression using analytic gradients and Hessians of the leave-one-out cross-validation (LOOCV) risk. It also supports other bandwidth selection methods such as grid search, plug-in rules, finite-difference Newton, golden-section search, and Bayesian optimisation.

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
