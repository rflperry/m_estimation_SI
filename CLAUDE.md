# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research package for **selective inference (SI) on penalized M-estimators** — constructing confidence intervals and hypothesis tests with valid coverage guarantees after data-driven variable selection via LASSO/group-LASSO in GLMs (linear and logistic regression).

## Setup

```bash
virtualenv env -p python3.10
source env/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/regreg/regreg.git
pip install -e .  # editable mode for development
pip install -r dev-requirements.txt  # adds pytest
```

Note: `pip install .` typically raises an error but installs successfully regardless.

For the RSC/PoSI comparison methods: locally clone `github.com/yiling-h/PoSI-GroupLASSO` and replace any `np.bool` with `bool`.

## Commands

**Run tests:**
```bash
pytest tests/
```

**Run a single test:**
```bash
pytest tests/test_logistic_glm.py::test_logistic_matches_builtin
```

**Debug a single experiment replicate:**
```bash
python experiments/glm_confints.py \
  --p 40 --level 0.90 --n 100 --n_reps 1 \
  --gamma 1 --sparsity 10 --signal 1 --lam 0.02 \
  --verbose --debug --family linear --error_model clustered --cluster_size 5
```

**Run a full experiment (background, logged):**
```bash
python experiments/glm_confints.py \
  --p 40 --level 0.90 --n 100 200 500 1000 --n_reps 1000 \
  --gamma 1 --sparsity 10 --lam 0.02 --n_jobs 4 \
  --family linear --error_model homogeneous \
  > logs/glm_confint_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

**Plot results:**
```bash
python experiments/plot_coverage.py \
  --fname results/<result_file.csv> --out figures/<output.png> --logx
```

See `run.sh` for canonical experiment configurations.

## Architecture

### Packages

- **`m_estimation_SI/`** — Core GLM fitting package (installed, importable)
  - `losses.py`: `logistic_loss_smooth` and `least_squares_loss_smooth` wrap regreg loss objects, exposing Hessian/variance computations
  - `penalized_glm.py`: `GLM` class — fits penalized GLMs via regreg, computes HC1/CR1 standard errors and confidence intervals
  - `simulation.py`: `logistic_group_instance()` generates synthetic data with configurable correlation structure, sparsity, and group structure

- **`selectinf/`** — Selective inference framework (not installed; imported by path)
  - `base.py`: `restricted_estimator()`, `selected_targets()` — core functions for constructing inference targets from a selection event
  - `randomization.py`: Randomization distributions (Gaussian, Laplace, etc.) for randomized LASSO
  - `group_lasso_query.py`: `group_lasso` class — main SI implementation for group LASSO; `fit()` solves the randomized problem, `form_targets()` / `pivot()` produce CIs and p-values
  - `group_lasso_query_quasi.py`: Quasi-likelihood variant for non-exponential family models

- **`selectinf2/`** — Alternative SI implementation for standard LASSO (not installed; imported by path)
  - `lasso.py`: LASSO-specific selective inference
  - `grid_inference.py`, `exact_reference.py`: Grid-based and exact inference approaches

### Experiments

**`experiments/glm_confints.py`** is the primary experiment script. It:
1. Loops over sample sizes `n` and Monte Carlo replicates
2. Generates data via `logistic_group_instance()`
3. Fits a penalized GLM, runs each SI method, records coverage/length/bias/TPR/FDR
4. Saves results as a CSV to `results/` with a filename encoding all parameters

Methods compared: `classic`, `sample_splitting`, `thin_gradient`, `thin_outcomes`, `rsc`, `rsc_exact`.

Results CSVs are named with the full parameter signature, e.g.:
`results/glm_confints_p=40_level=0.9_n=100,200,500,1000_reps=1000_gamma=1.0_s=10.0_lam=0.02_fam=linear_errors=homogeneous_mis=None_cluster_size=None_signal=1_dispersion=1_true_noise_var=False.csv`

### Key Design Patterns

- **regreg integration**: All optimization goes through regreg (`rr.simple_problem`, `rr.glm.logistic`, `rr.weighted_l1norm`). Loss functions in `losses.py` subclass regreg's `smooth_atom`.
- **Error models**: `homogeneous` uses HC1 SEs; `clustered` uses CR1 cluster-robust SEs with correction factor `(G/(G-1)) * ((n-1)/(n-p))`; `heterogeneous` uses per-observation variances.
- **`numpy<1.25`** constraint is hard — regreg breaks with newer numpy.
