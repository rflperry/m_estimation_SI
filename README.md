# m_estimation_SI

Selective inference for penalized M-estimators in Python.

After selecting variables via a penalized regression (e.g. lasso or group
lasso), standard confidence intervals are invalid because the data were used
twice — once to select and once to estimate. This package provides tools for
constructing confidence intervals that remain valid after data-driven variable
selection, and supports realistic error structures including clustered and
heteroskedastic observations.

## Quick start

```python
import numpy as np
from m_estimation_SI import GLM

rng = np.random.default_rng(0)
X = rng.standard_normal((300, 20))
beta_true = np.zeros(21)
beta_true[1:4] = [1.5, -1.0, 0.8]          # three active features
eta = np.c_[np.ones(300), X] @ beta_true
y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)

# Fit a penalised logistic regression
lam = np.sqrt(2 * np.log(20) / 300)
glm = GLM(family="logistic", l1_penalty=lam).fit(X, y)

print("Selected features:", glm.active())       # zero-indexed, excl. intercept
print("95% Wald CIs:\n", glm.conf_int(X))       # shape (21, 2)

# Cluster-robust standard errors (10 clusters of 30)
clusters = np.repeat(np.arange(10), 30)
print("CR1 CIs:\n", glm.conf_int(X, clusters=clusters))
```

## Installation

Create and activate a Python 3.10 virtual environment:

```bash
virtualenv env -p python3.10
source env/bin/activate
pip install -r requirements.txt
```

Install `regreg` from GitHub (required; not on PyPI):

```bash
pip install git+https://github.com/regreg/regreg.git
```

Install this package:

```bash
pip install .
```

> **Note:** `pip install .` may print an error message but installs successfully.

To also run the randomized conditional selective inference (RSC) comparison
method of Huang et al. (2025), locally clone and install
`github.com/yiling-h/PoSI-GroupLASSO`.  You may need to replace
`np.bool` with `bool` in its source.

## Core API

### `GLM(family, l1_penalty, ...)`

Penalized generalized linear model with robust sandwich standard errors.

| Argument | Description |
| --- | --- |
| `family` | `'linear'` or `'logistic'` |
| `l1_penalty` | Lasso penalty weight λ (mean-scaled; comparable across sample sizes) |
| `intercept` | Whether to fit an intercept (default `True`, never penalized) |
| `affine_penalty` | Additive linear term for randomized selective-inference objectives |

Key methods after `.fit(X, y)`:

| Method | Returns |
| --- | --- |
| `.predict(X)` | Fitted probabilities or values |
| `.active()` | Indices of selected features |
| `.conf_int(X, level, clusters)` | Wald CIs (HC1 or CR1 robust) |
| `.get_var(X, Y, error_model, clusters)` | Working variance estimates |

### `logistic_group_instance(n, p, sgroup, ...)`

Generates synthetic logistic regression data for group-lasso experiments.
Returns `(X, Y, beta_true, active_indices, sigma_X)`.

## Develop

Install development dependencies and the package in editable mode:

```bash
pip install -r dev-requirements.txt
pip install -e .
```

## Testing

Run all tests:

```bash
pytest tests/
```

Run a specific test file or test:

```bash
pytest tests/test_glm.py
pytest tests/test_glm.py::TestConfInt::test_lower_leq_upper
```

## References

- Huang, Y. et al. (2025). Randomized conditional selective inference for
  group lasso. *Preprint*.
- MacKinnon, J. G. & White, H. (1985). Some heteroskedasticity-consistent
  covariance matrix estimators. *Journal of Econometrics*, 29, 305–325.
- Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
  cluster-robust inference. *Journal of Human Resources*, 50, 317–372.
