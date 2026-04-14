# m_estimation_SI

Selective inference for penalized M-estimators in Python.

After selecting variables via a penalized regression (e.g. lasso), standard confidence intervals are invalid because the data were used
twice, once for selection and once for estimation. This package provides tools for constructing confidence intervals that remain valid after data-driven variable selection, and supports clustered and heteroskedastic errors.

## Quick start

The core workflow is **outcome thinning**: split the information in the
responses into a training half (for selection) and a testing half (for
inference) by adding and subtracting Gaussian noise whose variance matches
the estimated outcome variance.

```python
import numpy as np
from m_estimation_SI import GLM

rng = np.random.default_rng(0)
n, p = 300, 20
X = rng.standard_normal((n, p))
beta_true = np.zeros(p + 1)
beta_true[1:4] = [1.5, -1.0, 0.8]
eta = np.c_[np.ones(n), X] @ beta_true
Y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)

# 1. Estimate per-observation outcome variance with an unpenalised fit
glm_init = GLM(family="logistic").fit(X, Y)
Y_var = glm_init.get_var(X, Y, error_model="heterogeneous")

# 2. Draw noise scaled by the estimated variance and split the outcomes
gamma = 1.0                              # controls train/test information ratio
W = rng.normal(0, np.sqrt(Y_var))
Y_train = Y + gamma * W                 # used for variable selection
Y_test  = Y - W / gamma                 # used for inference

# 3. Select features on the training outcomes
lam = 0.05
glm_sel = GLM(family="logistic", l1_penalty=lam).fit(X, Y_train)
selected = glm_sel.active()             # zero-indexed, excludes intercept
print("Selected features:", selected)

# 4. Refit on the selected features using the testing outcomes
X_sel = X[:, selected]
glm_inf = GLM(family="logistic").fit(X_sel, Y_test)
ci = glm_inf.conf_int(X_sel, level=0.95)
print("95% confidence intervals (intercept + selected features):\n", ci)
```

A complete worked example on real friendship-network data is in
[glasgow_analysis.ipynb](glasgow_analysis.ipynb).

## Installation

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install git+https://github.com/regreg/regreg.git
uv pip install .
```

Or with pip and virtualenv:

```bash
virtualenv env -p python3.10
source env/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/regreg/regreg.git
pip install .
```

> **Note:** The install step may print an error but completes successfully.

To use the randomized conditional selective inference (RSC) comparison method
of Huang et al. (2025), locally clone and install
`github.com/yiling-h/PoSI-GroupLASSO`.  You may need to replace `np.bool`
with `bool` in its source.

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
uv pip install -r dev-requirements.txt
uv pip install -e .
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

- Neufeld, A. et al. (2024). Cohort selection and post-selection inference
  via data thinning. *Preprint*.
- Huang, Y. et al. (2025). Randomized conditional selective inference for
  group lasso. *Preprint*.
- MacKinnon, J. G. & White, H. (1985). Some heteroskedasticity-consistent
  covariance matrix estimators. *Journal of Econometrics*, 29, 305–325.
- Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
  cluster-robust inference. *Journal of Human Resources*, 50, 317–372.
