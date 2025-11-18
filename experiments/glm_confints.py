# %%
import numpy as np
import argparse
import os
import datetime
from m_estimation_SI import GLM
import pandas as pd
from joblib import Parallel, delayed
from m_estimation_SI.simulation import logistic_group_instance

import regreg.api as rr
from selectinf.group_lasso_query import group_lasso
from selectinf.base import selected_targets
import itertools

HEADER = [
    "rep",
    "n",
    "p",
    "sparsity",
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
    reference_set = (
        set(map(int, reference.split(","))) if reference is not None else set()
    )
    estimate_set = set(map(int, estimate.split(","))) if estimate is not None else set()

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


def classic_si(family, X, Y, mu, penalty, level, intercept):
    sel = GLM(family=family, l1_penalty=penalty, intercept=intercept).fit(X, Y).active()

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X[:, sel]
        model = GLM(family=family, intercept=intercept).fit(X_sel, Y)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, Y_var=None, level=level)
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


def sample_splitting_si(family, X, Y, mu, Y_var, penalty, level, gamma, intercept):
    n = X.shape[0]
    train_indices = np.random.choice(n, int(n * 1 / (1 + gamma)), replace=False)
    test_indices = np.setdiff1d(np.arange(n), train_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=intercept)
        .fit(X_train, Y_train)
        .active()
    )

    if Y_var is not None:
        Y_var = Y_var[test_indices]

    if len(sel) == 0:
        return [0, 0, 0, 0, 0, None]
    else:
        X_sel = X_test[:, sel]
        model = GLM(family=family, intercept=intercept).fit(X_sel, Y_test)
        beta_est = model.beta_
        conf_int = model.conf_int(X_sel, Y_var=None, level=level)
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


def thin_outcomes_si(family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model):

    if Y_var is not None:
        W *= np.sqrt(Y_var)
    elif X.shape[0] > X.shape[1] + 1:  # n > p
        Y_var_noise = (
            GLM(
                family=family,
                intercept=True
            )
            .fit(X, Y)
            .get_var(X, Y, error_model)
        )

        W *= np.sqrt(Y_var_noise)
    else:  # p > n
        Y_var_noise = (
            GLM(
                family=family,
                intercept=True,
                l1_penalty=penalty,
            )
            .fit(X, Y)
            .get_var(X, Y, error_model)
        )

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
        # model.residuals_ = Y - model.predict(X_sel) # non-noisy residuals for sandwich SE
        conf_int = model.conf_int(
            X_sel, Y_var=None, level=level, var_scale=1  # + 1 / gamma**2
        )  # correct for noisy penalty
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


def thin_gradient_si(family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model):

    if intercept:
        # Because X is used to form the penalty, we need to include intercept here
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    n, p = X.shape

    if Y_var is not None:
        W *= np.sqrt(Y_var)
    elif n > p:  # n > p
        Y_var_noise = (
            GLM(
                family=family,
                intercept=False
            )
            .fit(X, Y)
            .get_var(X, Y, error_model)
        )

        W *= np.sqrt(Y_var_noise)
    else:  # p > n
        Y_var_noise = (
            GLM(
                family=family,
                intercept=False,
                l1_penalty=penalty,
            )
            .fit(X, Y)
            .get_var(X, Y, error_model)
        )

        W *= np.sqrt(Y_var_noise)

    sel = (
        GLM(
            family=family,
            intercept=False,
            l1_penalty=penalty,
            affine_penalty=gamma * X.T @ W / np.sqrt(n),
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
        affine_penalty=-X_sel.T @ W / gamma / np.sqrt(X.shape[0]),
    ).fit(X_sel, Y)
    beta_est = model.beta_
    conf_int = model.conf_int(
        X_sel, Y_var=None, level=level, var_scale=1 + 1 / gamma**2
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
    family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model
):
    """Code adapted from github.com/yiling-h/PoSI-GroupLASSO"""

    if intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    n, p = X.shape

    if family == "linear":        
        if Y_var is not None:
            W *= np.sqrt(Y_var)
            hess = X.T @ np.diag(Y_var) @ X * gamma
        elif n > p:  # n > p
            Y_var_noise = (
                GLM(
                    family=family,
                    intercept=False
                )
                .fit(X, Y)
                .get_var(X, Y, error_model)
            )

            W *= np.sqrt(Y_var_noise)
            hess = X.T @ np.diag(Y_var_noise) @ X * gamma
        else:  # p > n
            Y_var_noise = (
                GLM(
                    family=family,
                    intercept=False,
                    l1_penalty=penalty,
                )
                .fit(X, Y)
                .get_var(X, Y, error_model)
            )

            W *= np.sqrt(Y_var_noise)
            hess = X.T @ np.diag(Y_var_noise) @ X * gamma

        groups = np.arange(p)
        weights = dict([(i, penalty) for i in np.unique(groups)])
        perturb = -gamma * X.T @ W / np.sqrt(n)  # they subtract the noise

        # By setting useJacobian=False, we
        conv = group_lasso.gaussian(
            X=X, Y=Y, groups=groups, weights=weights, perturb=perturb, useJacobian=False, cov_rand=hess
        )
        # cov_rand=hess)
        conv.useJacobian = True  # enabling Jacobian during initialization alters selection, this agrees with thin_gradients
        signs, _ = conv.fit()
        sel = np.where(signs != 0)[0]

        if len(sel) == 0:
            return [0, 0, 0, 0, 0, None]

        conv.setup_inference(dispersion=1)

        target_spec = selected_targets(
            conv.loglike, conv.observed_soln, dispersion=1
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
    else:
        raise ValueError(f"Family {family} not supported")


def run_simulation(
    n,
    p,
    family,
    gamma=1,
    lam=1,
    signal_frac=0.1,
    sparsity=3,
    rho=0.5,
    level=0.90,
    var_inflation=1.0,
    verbose=False,
    true_noise_var=False,
    intercept=False,
    error_model=None
):
    if family == "logistic":
        inst = logistic_group_instance
    elif family == "linear":
        inst = logistic_group_instance
    else:
        raise ValueError("Unsupported family")

    signal = 1 # np.sqrt(signal_frac * 2 * np.log(p))
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
    if family == "logistic":
        eta = X @ beta_true
        mu = 1 / (1 + np.exp(-eta))
        Y = np.random.binomial(1, mu, size=n)
        if true_noise_var:
            Y_var = mu * (1 - mu) * var_inflation
    elif family == "linear":
        mu = X @ beta_true
        Y = np.random.normal(mu, np.sqrt(var_inflation), size=n)
        if true_noise_var:
            Y_var = np.ones(X.shape[0]) * var_inflation
    else:
        raise ValueError("Unsupported family")

    # L1 penalty
    penalty = lam * np.sqrt(2 * np.log(p) / n) * np.std(Y)

    # Store results
    results = {}

    # ----- Classical -----
    try:
        results["classic"] = classic_si(
            family, X, Y, mu, penalty, level, intercept
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
            family, X, Y, mu, Y_var, penalty, level, gamma, intercept
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                results["classic"][-3], results["sample_splitting"][-1]
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
            family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                results["classic"][-3], results["thin_outcomes"][-1]
            )
            results["thin_outcomes"] += [TPR, FDR]
        else:
            results["thin_outcomes"] += [None, None]
    except Exception as e:
        print(f"Error in thin_outcomes: {e}")
        results["thin_outcomes"] = [None] * 8

    # ----- Thinning gradient -----
    try:
        results["thin_gradient"] = thin_gradient_si(
            family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(
                results["classic"][-3], results["thin_gradient"][-1]
            )
            results["thin_gradient"] += [TPR, FDR]
        else:
            results["thin_gradient"] += [None, None]
    except Exception as e:
        print(f"Error in thin_gradient: {e}")
        results["thin_gradient"] = [None] * 8

    # ----- Randomized Conditional (Huang 2025) -----
    try:
        results["rsc"] = randomized_conditional(
            family, X, Y, mu, Y_var, penalty, level, gamma, W, intercept, error_model
        )
        if results["classic"][-1] is not None:
            TPR, FDR = selection_accuracy(results["classic"][-3], results["rsc"][-1])
            results["rsc"] += [TPR, FDR]
        else:
            results["rsc"] += [None, None]
    except Exception as e:
        print(f"Error in RSC: {e}")
        results["rsc"] = [None] * 8

    return results


# %%


def main(args):
    # encode hyperparameters into a filename
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Arguements:, {args}")
    tag = (
        f"p={','.join(map(str,args.p))}_level={args.level}_n={','.join(map(str,args.n))}"
        f"_reps={args.n_reps}_gamma={args.gamma}_s={','.join(map(str,args.sparsity))}"
        f"_lam={args.lam}_fam={args.family}_errors={args.error_model}_true_noise_var={args.true_noise_var}"
    )
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    outfile = os.path.join(OUT_PATH, f"{script_name}_{tag}.csv")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    first_write = True
    grid = list(itertools.product(args.n, args.p, args.sparsity))
    for n, p, sparsity in grid:
        print(f"Running: n={n}, p={p}, sparsity={sparsity}")
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(run_simulation)(
                n=n,
                p=p,
                family=args.family,
                gamma=args.gamma,
                sparsity=sparsity,
                level=args.level,
                lam=args.lam,
                var_inflation=args.var_inflation,
                true_noise_var=args.true_noise_var,
                intercept=args.intercept,
                error_model=args.error_model
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
                        [dict(zip(HEADER, [rep, n, p, sparsity, method] + r))]
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
    parser.add_argument("--lam", type=float)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--family", type=str)
    parser.add_argument("--error_model", default=None, type=str)
    parser.add_argument("--var_inflation", type=int, default=1)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--intercept", default=False, action="store_true")
    parser.add_argument("--true_noise_var", default=False, action="store_true")
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
