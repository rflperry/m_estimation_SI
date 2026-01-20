# %%
import numpy as np
import argparse
import os
import datetime
from m_estimation_SI import GLM
import pandas as pd
from joblib import Parallel, delayed
from m_estimation_SI.simulation import logistic_group_instance

from selectinf.group_lasso_query import group_lasso

# from selectinf2.Tests.instance import gaussian_instance
from selectinf2.lasso import lasso
from selectinf2.Utils.base import selected_targets as selected_targets2

from selectinf.base import selected_targets
import itertools

HEADER = [
    "rep",
    "n",
    "p",
    "sparsity",
    "dispersion",
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


def classic_si(family, X, Y, mu, penalty, level, intercept, clusters=None):
    sel = GLM(family=family, l1_penalty=penalty, intercept=intercept).fit(X, Y).active()

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X[:, sel]
        model = GLM(family=family, intercept=intercept).fit(X_sel, Y)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, level=level, clusters=clusters)
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = GLM(family=family, intercept=intercept).fit(X_sel, mu).beta_
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
    family, X, Y, mu, Y_var, penalty, level, gamma, intercept, clusters=None
):
    n = X.shape[0]

    if clusters is None:
        train_indices = np.random.choice(n, int(n * 1 / (1 + gamma)), replace=False)
    else:
        unique_clusters = np.unique(clusters)
        train_clusters = np.random.choice(
            unique_clusters, size=int(len(unique_clusters) * 0.5), replace=False
        )
        train_indices = np.where(np.isin(clusters, train_clusters))[0]

    test_indices = np.setdiff1d(np.arange(n), train_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept)
        .fit(X_train, Y_train)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X_test[:, sel]
        model = GLM(family=family, intercept=intercept).fit(X_sel, Y_test)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, level=level)
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = (
            GLM(family=family, intercept=intercept).fit(X_sel, mu[test_indices]).beta_
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
):

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
        W = block_sqrt_mult(W, Y_var_noise)
    else:
        W *= np.sqrt(Y_var_noise)

    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept)
        .fit(X, Y + W * gamma)
        .active()
    )

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X[:, sel]
        model = GLM(family=family, intercept=intercept).fit(X_sel, Y - W / gamma)
        beta_est = model.beta_
        if true_sandwich:
            conf_int = model.conf_int(X_sel, level=level, clusters=clusters)
        else:
            raise ValueError("not implemented")
        length = conf_int[:, 1] - conf_int[:, 0]
        beta_sel = GLM(family=family, intercept=intercept).fit(X_sel, mu).beta_
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

    perturb = gamma * X.T @ W  # they subtract the noise, loss is not scaled by 1/n

    if family == "linear":
        conv = lasso.gaussian(
            X,
            Y,
            penalty * n,
            ridge_term=0.,
            # randomizer_scale=hess,
        )
    elif family == "logistic":
        conv = lasso.logistic(
            X,
            Y,
            penalty * n,
            ridge_term=0.,
            # randomizer_scale=hess,
        )
        dispersion = 1
    else:
        raise ValueError(f"Family {family} not supported")

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
):
    if family == "logistic":
        inst = logistic_group_instance
    elif family == "linear":
        inst = logistic_group_instance
    else:
        raise ValueError("Unsupported family")

    # signal = np.sqrt(signal_frac * 2 * np.log(p))
    if sparsity < 1:
        sparsity = int(sparsity * p)
    else:
        sparsity = int(sparsity)

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
    else:
        raise ValueError("Unsupported family")

    # L1 penalty
    penalty = lam * np.sqrt(2 * np.log(p) / n) * np.std(Y)

    # Store results
    results = {}

    # ----- Classical -----
    try:
        results["classic"] = classic_si(
            family, X.copy(), Y, mu, penalty, level, intercept, clusters=clusters
        )
        TPR, FDR = selection_accuracy(
            ",".join(map(str, np.where(beta_true != 0)[0])), results["classic"][-1]
        )
        results["classic"] += [TPR, FDR]  # placeholders for TPR, FDR
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
            Y_var,
            np.sqrt(1 + gamma) * penalty,
            level,
            gamma,
            intercept,
            clusters=clusters,
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                # results["classic"][-3]
                ",".join(map(str, np.where(beta_true != 0)[0])),
                results["sample_splitting"][-1],
            )
            results["sample_splitting"] += [TPR, FDR]
        else:
            results["sample_splitting"] += [None, None]
    except Exception as e:
        print(f"Error in sample_splitting: {e}")
        results["sample_splitting"] = [None] * 8

    # ----- Thinning outcomes -----
    W = np.random.normal(0, 1, size=n)
    try:
        results["thin_outcomes"] = thin_outcomes_si(
            family,
            X.copy(),
            Y,
            mu,
            Y_var,
            np.sqrt(1 + gamma) * penalty,
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
    #         np.sqrt(1 + gamma) * penalty,
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
    if error_model == "clustered":
        return results

    try:
        results["rsc"] = randomized_conditional(
            family,
            X,
            Y,
            mu,
            Y_var,
            np.sqrt(1 + gamma) * penalty,
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
            np.sqrt(1 + gamma) * penalty,
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
        f"_reps={args.n_reps}_gamma={args.gamma}_s={','.join(map(str,args.sparsity))}"
        f"_lam={args.lam}_fam={args.family}_errors={args.error_model}_mis={args.misspecified}_cluster_size={cluster_label}"
        f"_signal={args.signal}"
        f"_dispersion={','.join(map(str,args.dispersion))}_true_noise_var={args.true_noise_var}"
    )
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    outfile = os.path.join(OUT_PATH, f"{script_name}_{tag}.csv")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    first_write = True
    grid = list(
        itertools.product(args.n, args.p, args.sparsity, args.dispersion, clusters)
    )
    for n, p, sparsity, dispersion, cluster_size in grid:
        print(
            f"Running: n={n}, p={p}, sparsity={sparsity}, dispersion={dispersion}, cluster_size={cluster_size}"
        )
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(run_simulation)(
                n=n,
                p=p,
                family=args.family,
                gamma=args.gamma,
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
            )
            for _ in range(args.n_reps)
        )
        if args.debug:
            print(results)
            continue

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
    parser.add_argument("--gamma", type=float, default=1)
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
    args = parser.parse_args()
    main(args)

# else:
#     args = argparse.Namespace(
#         p = 40,
#         level = 0.9,
#         n = [100],
#         n_reps = 4,
#         gamma = 1,
#         sparsity =5,
#         lam =5,
#         n_jobs =1,
#         family ="logistic",
#         verbose =True,
#     )
#     print(args)
#     main(args)
# p = 40
# level = 0.90
# n = [100, 200, 500, 1000]
# n_reps = 16
# gamma = 1
# sparsity = 5
# lam = 5
# family = "logistic"

# %%
