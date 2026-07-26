# %%
import numpy as np
import argparse
import os
import datetime
from m_estimation_SI import GLM
import pandas as pd
from joblib import Parallel, delayed
from m_estimation_SI.simulation import logistic_group_instance
from m_estimation_SI.utils.cv import (
    select_lambda_by_mse_for_lasso,
    select_lambda_lassocv,
)

from selectinf.group_lasso_query import group_lasso

# from selectinf2.Tests.instance import gaussian_instance
import regreg.api as rr
from selectinf2.lasso import lasso
from selectinf2.randomization import randomization as _randomization2
from selectinf2.Utils.base import selected_targets as selected_targets2

from selectinf.base import selected_targets
import itertools

HEADER = [
    "rep",
    "n",
    "p",
    "sparsity",
    "dispersion",
    "gamma",
    "cluster_size",
    "method",
    "sum_covers",
    "sum_rejects",
    "sum_bias",
    "num_selected",
    "sum_length",
    "model",
    "TPR",
    "FDR",
]
OUT_PATH = "results"


def selection_accuracy(reference, estimate):
    """Compute TPR and FDR of estimate compared to reference model."""
    if reference is None or reference == "":
        reference_set = set()
    else:
        reference_set = set(map(int, reference.split(",")))

    if estimate is None or estimate == "":
        estimate_set = set()
    else:
        estimate_set = set(map(int, estimate.split(",")))

    true_positives = len(reference_set & estimate_set)
    false_positives = len(estimate_set - reference_set)
    false_negatives = len(reference_set - estimate_set)

    TPR = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    FDR = (
        false_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    return TPR, FDR


def block_sqrt_mult(W, Y_var, Z=None):
    """
    Multiply each group of q elements of W by the matrix square root of Y_var.

    Parameters
    ----------
    W : array_like, shape (n,)
        Vector of length n = q * m where m is number of clusters.
    Y_var : array_like, shape (q, q)
        Covariance matrix for one cluster.
    Z : array_like, shape (n, p)

    Returns
    -------
    W_new : ndarray, shape (n,)
        Transformed vector where each block of length q has been replaced by
        Y_var_sqrt @ original_block.
    """
    W = np.asarray(W)
    Y_var = np.asarray(Y_var)
    n = W.shape[0]
    q = Y_var.shape[0]
    assert n % q == 0, "Length of W must be a multiple of q"
    m = n // q

    # Symmetric matrix square root via eigen-decomposition
    vals, vecs = np.linalg.eigh(Y_var)

    # Numerical floor for tiny negative eigenvalues due to rounding
    vals[vals < 0] = 0.0
    sqrt_vals = np.sqrt(vals)
    Y_sqrt = (vecs * sqrt_vals) @ vecs.T  # equals vecs @ diag(sqrt_vals) @ vecs.T

    # Apply blockwise
    W_new = np.empty_like(W, dtype=float)
    if Z is not None:
        Z_new = np.zeros((Z.shape[1], Z.shape[1]), dtype=float)
    for j in range(m):
        start = j * q
        stop = start + q
        block = W[start:stop]
        W_new[start:stop] = Y_sqrt @ block
        if Z is not None:
            Z_new += Z[start:stop, :].T @ Y_var @ Z[start:stop, :]

    if Z is not None:
        return W_new, Z_new
    else:
        return W_new


def classic_si(
    family, X, Y, mu, penalty, level, intercept, clusters=None, glm_kwargs=None
):
    if glm_kwargs is None:
        glm_kwargs = {}

    if penalty < 0:
        # penalty = select_lambda_by_mse_for_lasso(family, X, Y, intercept, -1 * penalty)
        penalty = select_lambda_lassocv(X, Y, intercept, base_penalty=-1 * penalty)

    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept, **glm_kwargs)
        .fit(X, Y)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X[:, sel]
        model = GLM(family=family, intercept=intercept, **glm_kwargs).fit(X_sel, Y)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, level=level, clusters=clusters)
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = (
            GLM(family=family, intercept=intercept, **glm_kwargs).fit(X_sel, mu).beta_
        )
        cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
        rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
        return [
            sum(cover),
            sum(rejects),
            sum(np.abs(beta_est - beta_sel)),
            len(sel) + intercept,
            sum(length),
            ",".join(map(str, sel)),
        ]


def sample_splitting_si(
    family, X, Y, mu, penalty, level, gamma, intercept, clusters=None, glm_kwargs=None
):
    if glm_kwargs is None:
        glm_kwargs = {}
    n = X.shape[0]

    if clusters is None:
        train_indices = np.random.choice(n, int(n * 1 / (1 + gamma**2)), replace=False)
    else:
        unique_clusters = np.unique(clusters)
        train_clusters = np.random.choice(
            unique_clusters, size=int(len(unique_clusters) * 0.5), replace=False
        )
        train_indices = np.where(np.isin(clusters, train_clusters))[0]

    test_indices = np.setdiff1d(np.arange(n), train_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    if penalty < 0:
        # penalty = select_lambda_by_mse_for_lasso(
        #     family, X_train, Y_train, intercept, -1 * penalty
        # )
        penalty = select_lambda_lassocv(
            X_train, Y_train, intercept, base_penalty=-1 * penalty
        )

    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept, **glm_kwargs)
        .fit(X_train, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X_test[:, sel]
        model = GLM(family=family, intercept=intercept, **glm_kwargs).fit(X_sel, Y_test)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, level=level)
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = (
            GLM(family=family, intercept=intercept, **glm_kwargs)
            .fit(X_sel, mu[test_indices])
            .beta_
        )
        cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
        rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
        return [
            sum(cover),
            sum(rejects),
            sum(np.abs(beta_est - beta_sel)),
            len(sel) + intercept,
            sum(length),
            ",".join(map(str, sel)),
        ]


def poisson_thinning_si(X, Y, mu, penalty, level, gamma, intercept):
    """Selective inference via Poisson data thinning (Neufeld & Dharamshi).

    Uses the Binomial splitting property of the Poisson distribution: if
    Y_i ~ Poisson(mu_i) then for epsilon in (0, 1),

      Y_thin[i] | Y[i]  ~  Binomial(Y[i], epsilon)
      Y_train[i]         =  Y[i] - Y_thin[i]

    are independent, with Y_train ~ Poisson((1-epsilon)*mu) and
    Y_thin ~ Poisson(epsilon*mu). Selection uses Y_train; inference uses
    Y_thin. Unlike sample splitting, all n observations contribute to both
    halves.

    epsilon = gamma**2 / (1 + gamma**2) mirrors the thin_outcomes_si convention,
    giving an information ratio inference:selection of gamma**2.

    Both halves are rescaled by their epsilon factors to restore mean mu,
    so no intercept offset adjustment is required and the intercept flag
    is respected as-is.
    """
    n, _ = X.shape
    epsilon = gamma**2 / (1 + gamma**2)

    # Binomial thinning — requires integer counts
    Y_int = np.round(Y).astype(int)
    Y_test = np.random.binomial(Y_int, epsilon).astype(float)

    # Rescale each half to have mean mu (quasi-likelihood fit remains consistent)
    Y_train = (Y - Y_test) / (1 - epsilon)
    Y_test /= epsilon

    if penalty < 0:
        penalty = select_lambda_by_mse_for_lasso(
            "poisson", X, Y_train, True, -1 * penalty
        )

    sel = (
        GLM(family="poisson", l1_penalty=penalty, intercept=intercept)
        .fit(X, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    X_sel = X[:, sel]
    model = GLM(family="poisson", intercept=intercept).fit(X_sel, Y_test)
    beta_est = model.beta_
    conf_int = model.conf_int(X_sel, level=level)
    length = conf_int[:, 1] - conf_int[:, 0]

    # Reference: unpenalized Poisson GLM on true means; intercept absorbs any
    # constant shift so slope coefficients match beta_true on the selected set.
    beta_sel = GLM(family="poisson", intercept=intercept).fit(X_sel, mu).beta_

    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def binomial_thinning_si(X, Y, mu, penalty, level, gamma, intercept):
    """Selective inference via binomial data thinning for Bernoulli/logistic data.

    Procedure:
      (i)   Group observations by unique feature vector; group k has m_k members.
      (ii)  Each observation in group k receives obs_weight = m_k / n.
      (iii) Thin each group's binomial count S_k via the hypergeometric:
              T_k | S_k ~ Hypergeometric(m_k, S_k, a_k),
            giving T_k ~ Binomial(a_k, p_k) and (S_k-T_k) ~ Binomial(b_k, p_k)
            independently.
      (iv)  Average within each split: Y_train[i] = T_k / a_k,
            Y_test[i] = (S_k - T_k) / b_k  (values in [0, 1]).
      (v)   Fit weighted logistic regression on Y_train for selection, then on
            Y_test for inference.

    Observations with a unique feature vector (m_k=1) cannot be split; a warning
    is issued and they are passed through unchanged into both halves with weight 1/n.
    """
    import warnings
    from scipy.stats import hypergeom

    n = X.shape[0]
    epsilon = gamma**2 / (1 + gamma**2)

    X_bin, group_ids = np.unique(X, axis=0, return_inverse=True)
    B = len(X_bin)

    Y_train = np.empty(B)
    Y_test = np.empty(B)
    obs_weights = np.empty(B)
    mu_bin = np.empty(B)

    singleton_warned = False
    for k in range(B):
        bin_idx = np.where(group_ids == k)[0]
        m_k = len(bin_idx)
        obs_weights[k] = m_k * B / n
        mu_bin[k] = mu[bin_idx[0]]

        if m_k == 1:
            if not singleton_warned:
                warnings.warn(
                    "Unique feature vectors found in binomial thinning; those "
                    "observations are kept in both train and test sets unchanged.",
                    stacklevel=2,
                )
                singleton_warned = True
            Y_train[k] = Y[bin_idx[0]]
            Y_test[k] = Y[bin_idx[0]]
            continue

        a_k = max(1, min(int(np.round((1 - epsilon) * m_k)), m_k - 1))
        b_k = m_k - a_k

        S_k = int(np.round(Y[bin_idx]).sum())
        T_k = hypergeom.rvs(m_k, S_k, a_k)

        Y_train[k] = T_k / a_k
        Y_test[k] = (S_k - T_k) / b_k

    penalty = penalty / np.sqrt(B)
    if penalty < 0:
        penalty = select_lambda_lassocv(
            X_bin, Y_train, intercept, base_penalty=-1 * penalty
        )

    sel = (
        GLM(
            family="logistic",
            l1_penalty=penalty,
            intercept=intercept,
            obs_weights=obs_weights,
        )
        .fit(X_bin, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    X_sel = X_bin[:, sel]
    model = GLM(family="logistic", intercept=intercept, obs_weights=obs_weights).fit(
        X_sel, Y_test
    )
    beta_est = model.beta_
    conf_int = model.conf_int(X_sel, level=level)
    length = conf_int[:, 1] - conf_int[:, 0]

    beta_sel = (
        GLM(family="logistic", intercept=intercept, obs_weights=obs_weights)
        .fit(X_sel, mu_bin)
        .beta_
    )
    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def logistic_fission_si(X, Y, mu, penalty, level, gamma, intercept, true_mu=False):
    """Selective inference via Logistic Fission (Discussion, Neufeld 2025).
    Epsilon = gamma / (1 + gamma) mirrors the sample-splitting convention.
    """
    n, p = X.shape

    if true_mu:
        mu_hat = mu
    elif n > p + 1:
        mu_hat = GLM(family="logistic", intercept=True).fit(X, Y).predict(X)
    else:  # p > n
        mu_hat = (
            GLM(family="logistic", intercept=True, l1_penalty=penalty)
            .fit(X, Y)
            .predict(X)
        )

    # epsilon = gamma / (1 + gamma)
    epsilon = 0.5 + np.sqrt(
        np.maximum(0, 1 - 4 * mu_hat * (1 - mu_hat) * (1 + gamma))
    ) / np.abs(2 * (1 - 2 * mu_hat))
    epsilon[epsilon >= 1] = 0.95

    # epsilon = 1 - 0.2 * np.sqrt(100 / n)
    epsilon = 0.8

    from scipy.stats import bernoulli

    Y = np.round(Y).astype(int)
    Z = bernoulli.rvs(epsilon)
    Y_train = (1 - Z) * Y + Z * (1 - Y)
    Y_test = Y

    if penalty < 0:
        penalty = select_lambda_by_mse_for_lasso(
            "logistic", X, Y_train, True, -1 * penalty
        )

    sel = (
        GLM(family="logistic", l1_penalty=penalty, intercept=intercept)
        .fit(X, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    X_sel = X[:, sel]
    log_ratio = np.log(epsilon / (1 - epsilon))
    offset = (1 - 2 * Y_train) * log_ratio
    model = GLM(family="logistic", intercept=intercept, offset=offset).fit(
        X_sel, Y_test
    )
    beta_est = model.beta_
    conf_int = model.conf_int(X_sel, level=level)
    length = conf_int[:, 1] - conf_int[:, 0]

    beta_sel = GLM(family="logistic", intercept=intercept).fit(X_sel, mu).beta_

    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def negbinom_thinning_si(
    X, Y, mu, penalty, level, gamma, intercept, family, theta=None
):
    """Selective inference via NB data thinning (Dharamshi et al., Neufeld et al.).

    Uses the convolution-closure property of the NB family.  If
    Y_i ~ NB(mu_i, theta), draw

      Y_thin[i] | Y[i]  ~  BetaBinomial(Y[i], epsilon*theta, (1-epsilon)*theta)
      Y_train[i]         =  Y[i] - Y_thin[i]

    giving independent halves Y_thin ~ NB(epsilon*mu, epsilon*theta) and
    Y_train ~ NB((1-epsilon)*mu, (1-epsilon)*theta).  Both are then rescaled
    by their respective epsilon factors to recover responses with mean mu,
    so inference targets are identical to those of classic_si and no offset
    adjustment is needed.

    epsilon = gamma**2 / (1 + gamma**2) mirrors the thin_outcomes_si convention,
    giving an information ratio inference:selection of gamma**2.
    Both halves are rescaled by their epsilon factors to restore mean mu,
    so no intercept offset adjustment is required and the intercept flag
    is respected as-is.

    When ``theta`` is ``None`` (the default), it is estimated from the full
    data via an initial unpenalized (n > p) or penalized (n <= p) NB GLM
    fit with IRLS, analogous to how ``thin_outcomes_si`` estimates the
    working variance when ``Y_var`` is not supplied.  The estimated theta
    is then used for all subsequent thinning and fitting steps.
    """
    from scipy.stats import betabinom as _betabinom

    n, p = X.shape

    # Estimate theta from the data if not provided
    if theta is None:
        if n > p:
            theta = GLM(family="negative_binomial", intercept=True).fit(X, Y).theta_
        else:
            theta = (
                GLM(family="negative_binomial", intercept=True, l1_penalty=penalty)
                .fit(X, Y)
                .theta_
            )

    epsilon = gamma**2 / (1 + gamma**2)
    theta_thin = epsilon * theta
    theta_train = (1 - epsilon) * theta

    # Beta-Binomial thinning: Y_thin | Y ~ BetaBinomial(Y, epsilon*theta, (1-epsilon)*theta)
    Y_int = np.round(Y).clip(min=0).astype(int)
    Y_thin_raw = _betabinom.rvs(Y_int, theta_thin, theta_train).astype(float)
    Y_train_raw = Y - Y_thin_raw

    # Rescale each half to have mean mu (quasi-likelihood fit remains consistent)
    Y_train = Y_train_raw / (1 - epsilon)
    Y_thin = Y_thin_raw / epsilon

    if penalty < 0:
        penalty = select_lambda_by_mse_for_lasso(
            "negative_binomial", X, Y_train, True, -1 * penalty
        )

    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept)
        .fit(X, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    X_sel = X[:, sel]
    model = GLM(family=family, intercept=intercept).fit(X_sel, Y_thin)
    beta_est = model.beta_
    conf_int = model.conf_int(X_sel, level=level)
    length = conf_int[:, 1] - conf_int[:, 0]

    beta_sel = GLM(family=family, intercept=intercept, theta=theta).fit(X_sel, mu).beta_
    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def thin_outcomes_si(
    family,
    X,
    Y,
    mu,
    Y_var,
    penalty,
    level,
    gamma,
    W,
    intercept,
    error_model,
    true_sandwich=True,
    clusters=None,
    glm_kwargs=None,
):
    if glm_kwargs is None:
        glm_kwargs = {}

    # Use overdispersed variance for Poisson to account for extra-Poisson variation
    # in the noise scaling; for other families the model variance is already adequate.
    overdispersed = family == "poisson"

    if Y_var is not None:
        Y_var_noise = Y_var
    elif X.shape[0] > X.shape[1] + 1:  # n > p
        Y_var_noise = (
            GLM(family=family, intercept=True, **glm_kwargs)
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters, overdispersed=overdispersed)
        )
    else:  # p > n
        Y_var_noise = (
            GLM(family=family, intercept=True, l1_penalty=penalty, **glm_kwargs)
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters, overdispersed=overdispersed)
        )

    if clusters is not None:
        W = block_sqrt_mult(W, Y_var_noise)
    else:
        W *= np.sqrt(Y_var_noise)

    if penalty < 0:
        # penalty = select_lambda_by_mse_for_lasso(
        #     family, X, Y + W * gamma, intercept, -1 * penalty
        # )
        penalty = select_lambda_lassocv(
            X, Y + W * gamma, intercept, base_penalty=-1 * penalty
        )

    # Thinning with different noise
    # n, p = X.shape
    # if Y_var is not None:
    #     mu_hat = mu
    # elif n > p + 1:
    #     mu_hat = GLM(family="logistic", intercept=True).fit(X, Y).predict(X)
    # else:  # p > n
    #     mu_hat = (
    #         GLM(family="logistic", intercept=True, l1_penalty=penalty)
    #         .fit(X, Y)
    #         .predict(X)
    #     )

    # from scipy.stats import bernoulli

    # W = bernoulli.rvs(mu_hat) - mu_hat

    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept, **glm_kwargs)
        .fit(X, Y + W * gamma)
        .active()
    )

    if len(sel) == 0:
        return ([0, 0, 0, 0, 0, None], penalty)
    else:
        X_sel = X[:, sel]
        model = GLM(family=family, intercept=intercept, **glm_kwargs).fit(
            X_sel, Y - W / gamma
        )
        beta_est = model.beta_
        if true_sandwich:
            conf_int = model.conf_int(X_sel, level=level, clusters=clusters)
        else:
            raise ValueError("not implemented")
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = (
            GLM(family=family, intercept=intercept, **glm_kwargs).fit(X_sel, mu).beta_
        )
        cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
        rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
        return (
            [
                sum(cover),
                sum(rejects),
                sum(np.abs(beta_est - beta_sel)),
                len(sel) + intercept,
                sum(length),
                ",".join(map(str, sel)),
            ],
            penalty,
        )


def thin_gradient_si(
    family,
    X,
    Y,
    mu,
    Y_var,
    penalty,
    level,
    gamma,
    W,
    intercept,
    error_model,
    clusters=None,
):

    if intercept:
        # Because X is used to form the penalty, we need to include intercept here
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    n, p = X.shape

    if Y_var is not None:
        Y_var_noise = Y_var
    elif n > p:  # n > p
        Y_var_noise = (
            GLM(family=family, intercept=~intercept)
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )
    else:  # p > n
        Y_var_noise = (
            GLM(
                family=family,
                intercept=~intercept,
                l1_penalty=penalty,
            )
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )

    if clusters is not None:
        W = block_sqrt_mult(W, Y_var_noise)
    else:
        W *= np.sqrt(Y_var_noise)

    sel = (
        GLM(
            family=family,
            intercept=False,
            l1_penalty=penalty,
            affine_penalty=gamma * -X.T @ W / n,
        )
        .fit(X, Y)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    X_sel = X[:, sel]
    model = GLM(
        family=family,
        intercept=False,
        affine_penalty=X_sel.T @ W / gamma / n,
    ).fit(X_sel, Y)
    beta_est = model.beta_
    if clusters is not None:
        raise ValueError("Not implemented for clusters or heterogeneous errors")
    else:
        if type(Y_var_noise) is int:
            correction = Y_var_noise * X_sel.T @ X_sel / gamma**2 / n
        else:
            correction = X_sel.T @ np.diag(Y_var_noise) @ X_sel / gamma**2 / n
    conf_int = model.conf_int(
        X_sel, level=level, correction=correction, clusters=clusters
    )  # correct for noisy penalty
    length = conf_int[:, 1] - conf_int[:, 0]
    beta_sel = GLM(family=family, intercept=False).fit(X_sel, mu).beta_
    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def randomized_conditional(
    family,
    X,
    Y,
    mu,
    Y_var,
    penalty,
    level,
    gamma,
    W,
    intercept,
    error_model,
    clusters=None,
):
    """
    Code adapted from github.com/yiling-h/PoSI-GroupLASSO

    Note, their code uses the typical regreg loss scaling without 1/n.
    Thus, we scale W and the penalty by n accordingly.
    """

    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    n, p = X.shape

    if Y_var is not None:
        Y_var_noise = Y_var
    elif X.shape[0] > X.shape[1] + 1:  # n > p
        Y_var_noise = (
            GLM(family=family, intercept=True)
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )
    else:  # p > n
        Y_var_noise = (
            GLM(
                family=family,
                intercept=True,
                l1_penalty=penalty,
            )
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )

    if clusters is not None:
        W, hess = block_sqrt_mult(W, Y_var_noise, X)
        hess *= gamma**2
        dispersion = np.mean(np.diag(Y_var_noise))
    else:
        W *= np.sqrt(Y_var_noise)  # scale W
        hess = X.T @ np.diag(Y_var_noise) @ X * gamma**2
        dispersion = np.mean(Y_var_noise)

    groups = np.arange(p)
    weights = dict([(i, penalty * n) for i in np.unique(groups)])
    perturb = gamma * X.T @ W  # they subtract the noise, loss is not scaled by 1/n

    if family == "linear":
        conv = group_lasso.gaussian(
            X=X,
            Y=Y,
            groups=groups,
            weights=weights,
            perturb=perturb,
            useJacobian=True,
            cov_rand=hess,
        )
    elif family == "logistic":
        conv = group_lasso.logistic(
            X=X,
            successes=Y,
            groups=groups,
            weights=weights,
            perturb=perturb,
            useJacobian=True,
            cov_rand=hess,
        )
        dispersion = 1
    else:
        raise ValueError(f"Family {family} not supported")

    signs, _ = conv.fit(solve_args={"tol": 1.0e-8 * n, "min_its": 50})
    sel = np.where(signs != 0)[0]

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    conv.setup_inference(dispersion=dispersion)

    target_spec = selected_targets(
        conv.loglike, conv.observed_soln, dispersion=dispersion
    )

    result, _ = conv.inference(target_spec, method="selective_MLE", level=level)

    beta_est = result["unbiased"]
    conf_int = np.asarray(result[["lower_confidence", "upper_confidence"]])
    beta_sel = GLM(family=family, intercept=False).fit(X[:, sel], mu).beta_

    length = conf_int[:, 1] - conf_int[:, 0]
    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def randomized_conditional_exact(
    family,
    X,
    Y,
    mu,
    Y_var,
    penalty,
    level,
    gamma,
    W,
    intercept,
    error_model,
    clusters=None,
):
    """
    Code adapted from github.com/BakshiSoham/SI-MRT/tree/main

    Note, their code uses the typical regreg loss scaling without 1/n.
    Thus, we scale W and the penalty by n accordingly.
    """

    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    n, _ = X.shape

    if Y_var is not None:
        Y_var_noise = Y_var
    elif X.shape[0] > X.shape[1] + 1:  # n > p
        Y_var_noise = (
            GLM(family=family, intercept=True)
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )
    else:  # p > n
        Y_var_noise = (
            GLM(
                family=family,
                intercept=True,
                l1_penalty=penalty,
            )
            .fit(X, Y)
            .get_var(X, Y, error_model, clusters=clusters)
        )

    if clusters is not None:
        W, hess = block_sqrt_mult(W, Y_var_noise, X)
        hess *= gamma**2
        dispersion = np.mean(np.diag(Y_var_noise))
    else:
        W *= np.sqrt(Y_var_noise)  # scale W
        hess = X.T @ np.diag(Y_var_noise) @ X * gamma**2
        dispersion = np.mean(Y_var_noise)

    perturb = gamma * X.T @ W  # they subtract the noise, loss is not scaled by 1/n

    if family == "linear":
        loglike = rr.glm.gaussian(X, Y)
    elif family == "logistic":
        loglike = rr.glm.logistic(X, Y)
        dispersion = 1
    else:
        raise ValueError(f"Family {family} not supported")

    # Build lasso directly so the randomizer covariance matches Cov(perturb) = hess.
    # lasso.gaussian / lasso.logistic hardcode their own cov_rand (X.T@X or isotropic),
    # which does not vary with gamma, causing CI widths to be insensitive to gamma.
    conv = lasso(loglike, np.ones(X.shape[1]) * penalty * n, 0.0, _randomization2.gaussian(hess))

    signs = conv.fit(perturb=perturb, solve_args={"tol": 1.0e-8 * n, "min_its": 50})
    sel = np.where(signs != 0)[0]

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]

    conv.setup_inference(dispersion=dispersion)

    target_spec = selected_targets2(
        conv.loglike, conv.observed_soln, dispersion=dispersion
    )

    result = conv.inference(target_spec, level=level)

    # result = conv.summary(observed_target,
    #                       cov_target,
    #                       cov_target_score,
    #                       alternatives,
    #                       ndraw=5000,
    #                       burnin=1000,
    #                       compute_intervals=True)

    beta_est = result["target"]
    conf_int = np.asarray(result[["lower_confidence", "upper_confidence"]])
    beta_sel = GLM(family=family, intercept=False).fit(X[:, sel], mu).beta_

    length = conf_int[:, 1] - conf_int[:, 0]
    cover = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])
    rejects = (0 < conf_int[:, 0]) | (0 > conf_int[:, 1])
    return [
        sum(cover),
        sum(rejects),
        sum(np.abs(beta_est - beta_sel)),
        len(sel) + intercept,
        sum(length),
        ",".join(map(str, sel)),
    ]


def run_simulation(
    n,
    p,
    family,
    gamma=1,
    lam=1,
    # signal_frac=0.1,
    signal=1,
    sparsity=3,
    rho=0.5,
    level=0.90,
    dispersion=1.0,
    verbose=False,
    true_noise_var=False,
    intercept=False,
    error_model=None,
    misspecified=False,
    cluster_size=None,
    discrete=False,
):
    if family in ("logistic", "linear", "poisson", "negative_binomial"):
        inst = logistic_group_instance
    else:
        raise ValueError(f"Unsupported family: {family}")

    # signal = np.sqrt(signal_frac * 2 * np.log(p))
    if sparsity < 1:
        sparsity = int(sparsity * p)
    else:
        sparsity = int(sparsity)

    if discrete:
        n_unique = 50
        X, _, beta_true = inst(
            n=n,
            p=p,
            signal=signal,
            sgroup=sparsity,
            groups=np.arange(p),
            ndiscrete=0,
            nlevels=5,
            equicorrelated=False,
            scale=True,
            center=False,
            rho=rho,
            random_signs=True,
        )[:3]
        X = np.repeat(X[:n_unique], n // n_unique, 0)
    else:
        X, _, beta_true = inst(
            n=n,
            p=p,
            signal=signal,
            sgroup=sparsity,
            groups=np.arange(p),
            ndiscrete=0,
            nlevels=5,
            equicorrelated=False,
            scale=True,
            center=False,
            rho=rho,
            random_signs=True,
        )[:3]

    Y_var = None
    theta = None
    clusters = None

    if misspecified in ("mean", "both"):
        mu = 0.1 * np.log(1 + np.exp(X @ beta_true))
        # mu = (
        #     + 0.5 * (X[:, 0] ** 2 - X[:, 0].mean() ** 2)
        #     + np.sin(np.pi * X[:, 1] * X[:, 2])
        # )
    else:
        mu = X @ beta_true

    if family == "logistic":
        mu = 1 / (1 + np.exp(-mu))
        Y = np.random.binomial(1, mu, size=n)
        if true_noise_var:
            Y_var = mu * (1 - mu) * dispersion

        if error_model == "clustered":
            raise ValueError("clusters not implemented for logistic")

    elif family == "poisson":
        if misspecified in ("var", "both"):
            raise ValueError("Variance misspecification not implemented for Poisson")
        if error_model == "clustered":
            raise ValueError("Clustered errors not implemented for Poisson")
        # mu was the linear predictor (possibly softplus-transformed for mean misspecification);
        # apply the log link inverse to get Poisson means.
        mu = np.exp(mu)
        if dispersion > 1:
            nb_disp = mu / (dispersion - 1)
            lambda_i = np.random.gamma(shape=nb_disp, scale=1.0 / nb_disp)
            Y = np.random.poisson(mu * lambda_i).astype(float)
        else:
            Y = np.random.poisson(mu).astype(float)
        if true_noise_var:
            # Poisson variance equals mean. Otherwise,
            # the overdisped NB variance, mu + mu**2 / nb_disp.
            Y_var = mu * dispersion

    elif family == "negative_binomial":
        if misspecified in ("var", "both"):
            raise ValueError("Variance misspecification not implemented for NB")
        if error_model == "clustered":
            raise ValueError("Clustered errors not implemented for NB")
        if dispersion <= 1:
            raise ValueError("Dispersion parameter must be greater than 1")
        # Log link: mu = exp(linear predictor)
        mu = np.exp(mu)
        # NB via Gamma-Poisson mixture (supports non-integer dispersion)
        # Solve for: dispersion * mu = mu + mu**2 / np_disp
        # nb_disp = np.mean(mu) / (dispersion - 1)
        nb_disp = dispersion
        lambda_i = np.random.gamma(shape=nb_disp, scale=1.0 / nb_disp)
        Y = np.random.poisson(mu * lambda_i).astype(float)
        if true_noise_var:
            Y_var = mu + mu**2 / nb_disp  # NB variance
            theta = nb_disp

    elif family == "linear":
        if misspecified in ("var", "both"):
            errors = np.random.laplace(loc=0, scale=np.sqrt(dispersion / 2), size=n)
        else:
            errors = np.random.normal(0, np.sqrt(dispersion), size=n)

        if error_model == "clustered":
            num_clusters = n // cluster_size
            clusters = np.repeat(np.arange(num_clusters), cluster_size)
            clusters = np.hstack(
                [clusters, np.random.choice(num_clusters, n - len(clusters))]
            )
            cluster_effects = np.random.normal(
                0, np.sqrt(dispersion), size=num_clusters
            )

            Y = mu + (cluster_effects[clusters] + errors) / np.sqrt(
                2
            )  # keeps variance on same scale.

            if true_noise_var:
                Y_var = (
                    dispersion
                    * (
                        np.eye(cluster_size)
                        + np.ones(shape=(cluster_size, cluster_size))
                    )
                    / 2
                )
        else:
            Y = mu + errors
            if true_noise_var:
                Y_var = np.ones(X.shape[0]) * dispersion

    # L1 penalty
    penalty = lam * np.sqrt(2 * np.log(p) / n) * np.std(Y)

    # Family-specific extra kwargs forwarded to every GLM constructor
    # glm_kwargs = {"theta": dispersion} if family == "negative_binomial" else {}
    glm_kwargs = {"theta": None} if family == "negative_binomial" else {}

    # Store results
    results = {}

    # ----- Classical -----
    try:
        results["classic"] = classic_si(
            family,
            X.copy(),
            Y,
            mu,
            penalty,
            level,
            intercept,
            clusters=clusters,
            glm_kwargs=glm_kwargs,
        )
        TPR, FDR = selection_accuracy(
            ",".join(map(str, np.where(beta_true != 0)[0])), results["classic"][-1]
        )
        results["classic"] += [TPR, FDR]
    except Exception as e:
        print(f"Error in classic: {e}")
        results["classic"] = [None] * 8

    # ----- Sample splitting -----
    try:
        results["sample_splitting"] = sample_splitting_si(
            family,
            X.copy(),
            Y,
            mu,
            np.sqrt(1 + gamma**2) * penalty,
            level,
            gamma,
            intercept,
            clusters=clusters,
            glm_kwargs=glm_kwargs,
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["sample_splitting"][-1],
            )
            results["sample_splitting"] += [TPR, FDR]
        else:
            results["sample_splitting"] += [None, None]
    except Exception as e:
        print(f"Error in sample_splitting: {e}")
        results["sample_splitting"] = [None] * 8

    # ----- Logistic fission -----
    # if family == "logistic":
    #     try:
    #         results["logistic_fission"] = logistic_fission_si(
    #             X.copy(),
    #             Y,
    #             mu,
    #             np.sqrt(1 + gamma**2) * penalty,
    #             level,
    #             gamma,
    #             intercept,
    #             Y_var is not None,
    #         )
    #         TPR, FDR = selection_accuracy(
    #             ",".join(map(str, np.where(beta_true != 0)[0])),
    #             results["logistic_fission"][-1],
    #         )
    #         results["logistic_fission"] += [TPR, FDR]
    #     except Exception as e:
    #         print(f"Error in logistic_fission: {e}")
    #         results["logistic_fission"] = [None] * 8

    # ----- Poisson thinning (exact for Poisson; misspecified but valid for NB) -----
    if family in ("poisson", "negative_binomial"):
        try:
            results["poisson_thinning"] = poisson_thinning_si(
                X.copy(),
                Y,
                mu,
                np.sqrt(1 + gamma**2) * penalty,
                level,
                gamma,
                intercept,
            )
            TPR, FDR = selection_accuracy(
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["poisson_thinning"][-1],
            )
            results["poisson_thinning"] += [TPR, FDR]
        except Exception as e:
            print(f"Error in poisson_thinning: {e}")
            results["poisson_thinning"] = [None] * 8

    # ----- NB thinning (exact for NB via Beta-Binomial splitting) -----
    # if family == "negative_binomial":
    if family in ("poisson", "negative_binomial"):
        try:
            results["negbinom_thinning"] = negbinom_thinning_si(
                X.copy(),
                Y,
                mu,
                np.sqrt(1 + gamma**2) * penalty,
                level,
                gamma,
                intercept,
                family,
                theta,
            )
            TPR, FDR = selection_accuracy(
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["negbinom_thinning"][-1],
            )
            results["negbinom_thinning"] += [TPR, FDR]
        except Exception as e:
            print(f"Error in negbinom_thinning: {e}")
            results["negbinom_thinning"] = [None] * 8

    # ----- Binomial thinning (logistic + discrete flag only) -----
    if family == "logistic" and discrete:
        try:
            results["binomial_thinning"] = binomial_thinning_si(
                X.copy(),
                Y,
                mu,
                np.sqrt(1 + gamma**2) * penalty,
                level,
                gamma,
                intercept,
            )
            TPR, FDR = selection_accuracy(
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["binomial_thinning"][-1],
            )
            results["binomial_thinning"] += [TPR, FDR]
        except Exception as e:
            print(f"Error in binomial_thinning: {e}")
            results["binomial_thinning"] = [None] * 8

    # ----- Thinning outcomes -----
    W = np.random.normal(0, 1, size=n)
    try:
        results["thin_outcomes"], penalty_learned = thin_outcomes_si(
            family,
            X.copy(),
            Y,
            mu,
            Y_var,
            np.sqrt(1 + gamma**2) * penalty,
            level,
            gamma,
            W.copy(),
            intercept,
            error_model,
            clusters=clusters,
            glm_kwargs=glm_kwargs,
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["thin_outcomes"][-1],
            )
            results["thin_outcomes"] += [TPR, FDR]
        else:
            results["thin_outcomes"] += [None, None]
    except Exception as e:
        print(f"Error in thin_outcomes: {e}")
        results["thin_outcomes"] = [None] * 8

    # ----- Thinning gradient -----
    # try:
    #     results["thin_gradient"] = thin_gradient_si(
    #         family,
    #         X.copy(),
    #         Y,
    #         mu,
    #         Y_var,
    #         np.sqrt(1 + gamma**2) * penalty,
    #         level,
    #         gamma,
    #         W.copy(),
    #         intercept,
    #         error_model,
    #     )
    #     if results["classic"][-1] is not None:
    #         TPR, FDR = selection_accuracy(
    #             # results["classic"][-3]
    #             ",".join(map(str, np.where(beta_true != 0)[0])),
    #             results["thin_gradient"][-1],
    #         )
    #         results["thin_gradient"] += [TPR, FDR]
    #     else:
    #         results["thin_gradient"] += [None, None]
    # except Exception as e:
    #     print(f"Error in thin_gradient: {e}")
    #     results["thin_gradient"] = [None] * 8

    # ----- Randomized Conditional (Huang 2025) -----
    # RSC only supports linear and logistic families.
    if error_model == "clustered" or family not in ("linear", "logistic"):
        return results

    try:
        results["rsc"] = randomized_conditional(
            family,
            X,
            Y,
            mu,
            Y_var,
            penalty_learned,  # output of score thinning, if CV
            level,
            gamma,
            W.copy(),
            intercept,
            error_model,
            clusters=clusters,
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                # results["classic"][-3]
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["rsc"][-1],
            )
            results["rsc"] += [TPR, FDR]
        else:
            results["rsc"] += [None, None]
    except Exception as e:
        print(f"Error in RSC: {e}")
        results["rsc"] = [None] * 8

    # ----- Randomized Conditional Exact -----
    if error_model == "clustered" or family != "linear":
        return results

    try:
        results["rsc_exact"] = randomized_conditional_exact(
            family,
            X,
            Y,
            mu,
            Y_var,
            penalty_learned,  # output of score thinning, if CV
            level,
            gamma,
            W.copy(),
            intercept,
            error_model,
            clusters=clusters,
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                # results["classic"][-3]
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["rsc_exact"][-1],
            )
            results["rsc_exact"] += [TPR, FDR]
        else:
            results["rsc_exact"] += [None, None]
    except Exception as e:
        print(f"Error in RSC Exact: {e}")
        results["rsc_exact"] = [None] * 8

    return results


# %%
def main(args):
    # encode hyperparameters into a filename

    if args.epsilon:
        # args.gamma = [np.round(eps / (1 - eps), 2) for eps in args.epsilon]
        info_name = 'epsilon'
        info_split = args.epsilon
    else:
        info_name = 'gamma'
        info_split = args.gamma

    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Arguements:, {args}")
    if args.cluster_size is not None:
        cluster_label = ",".join(map(str, args.cluster_size))
        clusters = args.cluster_size
    else:
        cluster_label = None
        clusters = [None]
    tag = (
        f"p={','.join(map(str,args.p))}_level={args.level}_n={','.join(map(str,args.n))}"
        f"_reps={args.n_reps}_{info_name}={','.join(map(str,info_split))}_s={','.join(map(str,args.sparsity))}"
        f"_lam={args.lam}_fam={args.family}_errors={args.error_model}_mis={args.misspecified}_cluster_size={cluster_label}"
        f"_signal={args.signal}"
        f"_dispersion={','.join(map(str,args.dispersion))}_true_noise_var={args.true_noise_var}"
        f"_discrete={args.discrete}"
    )
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    outfile = os.path.join(OUT_PATH, f"{script_name}_{tag}.csv")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    first_write = True
    grid = list(
        itertools.product(
            args.n, args.p, args.sparsity, args.dispersion, info_split, clusters
        )
    )
    for n, p, sparsity, dispersion, info, cluster_size in grid:
        print(
            f"Running: n={n}, p={p}, sparsity={sparsity}, dispersion={dispersion}, {info_name}={info}, cluster_size={cluster_size}"
        )

        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(run_simulation)(
                n=n,
                p=p,
                family=args.family,
                gamma=np.sqrt(info / (1 - info)) if info_name == 'epsilon' else info,
                sparsity=sparsity,
                level=args.level,
                signal=args.signal,
                lam=args.lam,
                dispersion=dispersion,
                true_noise_var=args.true_noise_var,
                intercept=args.intercept,
                error_model=args.error_model,
                misspecified=args.misspecified,
                cluster_size=cluster_size,
                discrete=args.discrete,
            )
            for _ in range(args.n_reps)
        )
        if args.debug:
            print(results)
            continue

        HEADER[5] = info_name
        print(f"Writing to: {outfile}")
        for rep, res in enumerate(results):
            try:
                for method, r in res.items():
                    df = pd.DataFrame(
                        [
                            dict(
                                zip(
                                    HEADER,
                                    [
                                        rep,
                                        n,
                                        p,
                                        sparsity,
                                        dispersion,
                                        info,
                                        cluster_size,
                                        method,
                                    ]
                                    + r,
                                )
                            )
                        ]
                    )
                    if first_write:
                        df.to_csv(outfile, mode="w", header=True, index=False)
                        first_write = False
                    else:
                        df.to_csv(outfile, mode="a", header=False, index=False)
            except ValueError as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, nargs="+")
    parser.add_argument("--level", type=float, default=0.90)
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--n_reps", type=int)
    parser.add_argument("--gamma", type=float, default=[1], nargs="+")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        nargs="+",
        help="Information fraction for inference",
    )
    parser.add_argument("--sparsity", type=float, nargs="+")
    parser.add_argument("--signal", type=int, default=1)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--family", type=str)
    parser.add_argument("--error_model", default=None, type=str)
    parser.add_argument("--dispersion", type=int, default=[1], nargs="+")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--misspecified", default=None, type=str)
    parser.add_argument("--intercept", default=False, action="store_true")
    parser.add_argument("--true_noise_var", default=False, action="store_true")
    parser.add_argument("--cluster_size", type=int, default=None, nargs="+")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--discrete",
        default=False,
        action="store_true",
        help="Enable binomial data thinning for logistic family (groups by unique feature vector, splits via hypergeometric).",
    )
    args = parser.parse_args()
    main(args)
