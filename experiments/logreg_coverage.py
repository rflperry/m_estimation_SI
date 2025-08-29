# %%
import numpy as np
from m_estimation_SI import GLM
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
from m_estimation_SI.simulation import logistic_group_instance

# %%
###################################
# Simulation study
###################################


def plot_selected_coeffs(beta_sel, conf_int, sel, p, title=""):
    """
    Plot point estimates and confidence intervals for selected coefficients.

    Parameters
    ----------
    beta_sel : array, shape (k,)
        Estimated coefficients for selected features.
    conf_int : array, shape (k, 2)
        Confidence intervals (lower, upper).
    sel : array, shape (k,)
        Indices of selected coefficients (subset of [0, ..., p]).
    p : int
        Total number of parameters (including intercept).
    """
    _, ax = plt.subplots(figsize=(8, 5))

    # Plot point estimates
    ax.scatter(sel, beta_sel, color="blue", label="Estimand", zorder=3)

    # Plot confidence intervals
    for i, idx in enumerate(sel):
        ax.plot([idx, idx], conf_int[i], color="black", lw=1.5)

    # Add reference line at 0
    ax.axhline(0, color="red", linestyle="--", lw=1)

    ax.set_xticks(range(p))
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.legend()
    plt.show()


# %%
def run_simulation(
    n, p, family, gamma=1, lam=1, signal_frac=0.1, sparsity=3, rho=0.5, level=0.90, verbose=False
):
    if family == "logistic":
        inst = logistic_group_instance
    else:
        raise ValueError("Unsupported family")
    
    signal = np.sqrt(signal_frac * 2 * np.log(p))
    if isinstance(sparsity, float):
        sparsity = int(sparsity * p)

    X, _, beta_true = inst(
        n=n,
        p=p,
        signal=signal,
        sgroup=sparsity,
        groups=np.arange(p),
        ndiscrete=0,
        nlevels=5,
        equicorrelated=False,
        scale = True,
        rho=rho,
        random_signs=True,
    )[:3]

    # X /= X.sd(0)[None, :]
    # beta_true /= np.sqrt(n)
    eta = X @ beta_true
    prob = 1 / (1 + np.exp(-eta))
    Y = np.random.binomial(1, prob, size=n)
    Y_var_true = prob * (1 - prob)

    # L1 penalty
    penalty = lam * np.sqrt(2 * np.log(p) / n)

    # ----- Classical -----
    sel = GLM(family=family, l1_penalty=penalty, intercept=False).fit(X, Y).active()

    if len(sel) == 0:
        cover_classic = []
    else:
        X_sel = X[:, sel]
        conf_int = (
            GLM(family=family, intercept=False)
            .fit(X_sel, Y)
            .conf_int(X_sel, Y_var_true, level=level)
        )
        beta_sel = GLM(family=family, intercept=False).fit(X_sel, prob).beta_
        cover_classic = (beta_sel >= conf_int[:, 0]) & (beta_sel <= conf_int[:, 1])

        if verbose:
            plot_selected_coeffs(beta_sel, conf_int, sel, p + 1, "Classical")

    # ----- Thinning outcomes -----
    W = np.random.normal(0, np.sqrt(Y_var_true))
    sel = (
        GLM(family=family, l1_penalty=penalty, intercept=False)
        .fit(X, Y + W * gamma)
        .active()
    )

    if len(sel) == 0:
        cover_thin_outcomes = []
    else:
        X_sel = X[:, sel]
        conf_int = (
            GLM(family=family, intercept=False)
            .fit(X_sel, Y - W / gamma)
            .conf_int(X_sel, Y_var_true * (1 + 1 / gamma**2), level=level)
        )
        beta_sel = GLM(family=family, intercept=False).fit(X_sel, prob).beta_
        cover_thin_outcomes = (beta_sel >= conf_int[:, 0]) & (
            beta_sel <= conf_int[:, 1]
        )

        if verbose:
            plot_selected_coeffs(beta_sel, conf_int, sel, p + 1, "Thinning outcomes")

    # ----- Thinning gradient -----
    sel = (
        GLM(
            family=family,
            intercept=False,
            l1_penalty=penalty,
            affine_penalty=gamma * X.T @ W,
        )
        .fit(X, Y)
        .active()
    )

    if len(sel) == 0:
        cover_thin_gradient = []
    else:
        X_sel = X[:, sel]
        conf_int = (
            GLM(family=family, intercept=False, affine_penalty=-X_sel.T @ W / gamma)
            .fit(X_sel, Y)
            .conf_int(X_sel, Y_var_true * (1 + 1 / gamma**2), level=level)
        )
        beta_sel = GLM(family=family, intercept=False).fit(X_sel, prob).beta_
        cover_thin_gradient = (beta_sel >= conf_int[:, 0]) & (
            beta_sel <= conf_int[:, 1]
        )

        if verbose:
            plot_selected_coeffs(beta_sel, conf_int, sel, p + 1, "Thinning gradient")

    return [cover_classic, cover_thin_outcomes, cover_thin_gradient]


#%%
# if __name__ == "__main__":
# out = run_simulation(100, 20, family="logistic"), verbose

p = 40
level = 0.90
n_values = [100, 200, 500, 1000]
n_reps = 16
gamma = 1
sparsity = 5
lam = 5
family = "logistic"
methods = ["classical", "outcomes", "gradient"]
coverage = {n: {m: [] for m in methods} for n in n_values}

def run_instance(seed=None, **kwargs):
    np.random.seed(seed)
    try:
        return run_simulation(**kwargs)
    except:
        return None

        
for n in n_values:
    # run in parallel across reps

    results = Parallel(n_jobs=4, verbose=0)(
        delayed(run_instance)(n=n, p=p, family=family, gamma=gamma, sparsity=sparsity, level=level, lam=lam)
        for _ in range(n_reps)
    )

    n_success = 0
    for out in results:
        if out is None:  # in case your run_simulation raises error and you catch inside
            print('Simulation failed, skipping')
            continue
        try:
            for m, o in zip(methods, out):
                coverage[n][m].extend(o)
            n_success += 1
        except ValueError as e:
            print(e)

    print(f"n={n}, success rate: {n_success/n_reps:.2f}")

# reshape into long-form dataframe
rows = []
for n, vals in coverage.items():
    for method, cov in vals.items():
        rows.append({"n": n, "method": method, "coverage": np.mean(cov)})
df = pd.DataFrame(rows)

plt.figure(figsize=(7, 5))
sns.lineplot(data=df, x="n", y="coverage", hue="method", marker="o")

plt.axhline(level, color="red", linestyle="--", linewidth=1)
plt.xlabel("Sample size n")
plt.ylabel("Coverage")
plt.ylim(0, 1.05)
plt.legend()
plt.show()


# %%    

p = 40
level = 0.90
sparsity_values = [0, 5, 10, 20, 40]
n_reps = 16
gamma = 1
n = 200
lam = 5
family = "logistic"
methods = ["classical", "outcomes", "gradient"]
coverage = {s: {m: [] for m in methods} for s in sparsity_values}

def run_instance(seed=None, **kwargs):
    # np.random.seed(seed)
    try:
        return run_simulation(**kwargs)
    except:
        return None

        
for sparsity in sparsity_values:
    # run in parallel across reps

    results = Parallel(n_jobs=4, verbose=0)(
        delayed(run_instance)(n=n, p=p, family=family, gamma=gamma, sparsity=sparsity, level=level, lam=lam)
        for _ in range(n_reps)
    )

    n_success = 0
    for out in results:
        if out is None:  # in case your run_simulation raises error and you catch inside
            print('Simulation failed, skipping')
            continue
        try:
            for m, o in zip(methods, out):
                coverage[sparsity][m].extend(o)
            n_success += 1
        except ValueError as e:
            print(e)

    print(f"sparsity={sparsity}, success rate: {n_success/n_reps:.2f}")

# reshape into long-form dataframe
rows = []
for sparsity, vals in coverage.items():
    for method, cov in vals.items():
        rows.append({"sparsity": sparsity, "method": method, "coverage": np.mean(cov)})
df = pd.DataFrame(rows)

plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sparsity", y="coverage", hue="method")

plt.axhline(level, color="red", linestyle="--", linewidth=1)
plt.xlabel("Sparsity")
plt.ylabel("Coverage")
plt.ylim(0, 1.05)
plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.show()


# %%    
