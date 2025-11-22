# %%
import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from m_estimation_SI import GLM

# %%
def lasso_select(X, Y, penalty):
    """
    Fit lasso (no intercept) and return selected feature indices.
    """

    # lasso = linear_model.Lasso(alpha=penalty, fit_intercept=False).fit(X, Y)
    # beta_hat = lasso.coef_
    # selected = np.where(beta_hat != 0)[0]
    
    lasso = GLM(family="linear", l1_penalty=penalty, intercept=False).fit(X, Y)
    beta_hat = lasso.beta_
    selected = np.where(beta_hat != 0)[0]

    return selected 


def ols_ci(Xs, Y, mu, level=0.90):
    """
    Fit OLS (no intercept) on selected features and compute HC1 CIs.
    Returns:
        beta_hat: OLS estimates (k-dimensional)
        ci: (k x 2) matrix of confidence intervals [lower, upper]
    """

    n, k = Xs.shape

    # OLS estimate: (X'X)^(-1) X'Y
    XtX = Xs.T @ Xs
    XtY = Xs.T @ Y
    XtX_inv = np.linalg.inv(XtX)

    beta_hat = XtX_inv @ XtY
    beta_true = XtX_inv @ (Xs.T @ mu)

    # Residuals
    resid = Y - Xs @ beta_hat

    # HC1 robust covariance:
    # meat = X' diag(e_i^2) X
    Xdiag = Xs * (resid**2)[:, None]
    meat = Xs.T @ Xdiag
    V_hc0 = XtX_inv @ meat @ XtX_inv

    # HC1 correction: n/(n-k)
    V_hc1 = (n / (n - k)) * V_hc0

    se = np.sqrt(np.diag(V_hc1))

    # Confidence intervals
    alpha = 1 - level
    z = norm.ppf(1 - alpha / 2)

    lower = beta_hat - z * se
    upper = beta_hat + z * se

    ci = np.column_stack([lower, upper])

    return beta_hat, beta_true, ci


# %%

p = 40
n_values = [100, 200, 500]
n_reps = 500
gamma = 1
signal_frac = 1
sparsity = 10
rho = 0.5
lam = 0.01
Y_var = 1
level = 0.90

coverage_results = {}
width_results = {}

# Construct Sigma^{1/2} where Sigma_ij = rho^{|i-j|}
indices = np.arange(1, p + 1)
Sigma = rho ** np.abs(np.subtract.outer(indices, indices))
# R uses Sigma12 = (rho^{|i-j|})^(1/2)
Sigma12 = np.linalg.cholesky(Sigma)


for n in n_values:
    coverage_classic = []
    coverage_thin_outcomes = []
    coverage_sample_split = []
    width_classic = []
    width_thin_outcomes = []
    width_sample_split = []

    for rep in range(n_reps):
        # Simulate design matrix X
        X = np.random.normal(0, 1, size=(n, p)) @ Sigma12
        # scale(center = FALSE) means divide by column SD but no centering
        # Equivalent in Python:
        # col_sd = X.std(axis=0, ddof=1)
        # X = X / col_sd  # no centering
        X = X / np.sqrt(n)

        # Simulate true coefficients
        beta_true = np.zeros(p)

        signal = np.sqrt(signal_frac)
        k = int(np.ceil(sparsity)) + 1
        beta_true[:k] = np.linspace(signal, 0, k)

        # Randomly flip signs
        beta_true = beta_true * np.random.choice([-1, 1], size=p, replace=True)

        mu = X @ beta_true
        Y = np.random.normal(0, np.sqrt(Y_var), size=n) + mu

        # Lasso penalty
        penalty = lam * np.sqrt(2 * np.log(p) / n) * np.sqrt(Y_var)

        # ---------------------------------------
        # Your further simulation code goes here.
        # ---------------------------------------

        # CLassic
        selected = lasso_select(X, Y, penalty)

        if len(selected) == 0:
            continue

        Xs = X[:, selected]

        beta_hat, beta_sel_true, ci = ols_ci(Xs, Y, mu, level=level)
        coverage = (beta_sel_true >= ci[:, 0]) & (beta_sel_true <= ci[:, 1])
        widths = ci[:, 1] - ci[:, 0]
        coverage_classic.extend(coverage.tolist())
        width_classic.extend(widths.tolist())
        
        # Sample splitting
        n_half = n // 2
        X1, X2 = X[:n_half], X[n_half:]
        Y1, Y2 = Y[:n_half], Y[n_half:]
        mu1, mu2 = mu[:n_half], mu[n_half:]

        selected_ss = lasso_select(X1, Y1, penalty)

        if len(selected_ss) > 0:
            Xs2 = X2[:, selected_ss]
            beta_hat_ss, beta_sel_true_ss, ci_ss = ols_ci(Xs2, Y2, mu2, level=level)
            coverage_ss = (beta_sel_true_ss >= ci_ss[:, 0]) & (beta_sel_true_ss <= ci_ss[:, 1])
            widths_ss = ci_ss[:, 1] - ci_ss[:, 0]
            coverage_sample_split.extend(coverage_ss.tolist())
            width_sample_split.extend(widths_ss.tolist())
        
        # thin outcomes
        W = np.random.normal(0, np.sqrt(Y_var), size=n)
        selected_to = lasso_select(X, Y + W, penalty)

        if len(selected_to) == 0:
            continue

        Xs = X[:, selected_to]

        beta_hat, beta_sel_true, ci = ols_ci(Xs, Y - W, mu, level=level)
        coverage_to = (beta_sel_true >= ci[:, 0]) & (beta_sel_true <= ci[:, 1])
        widths_to = ci[:, 1] - ci[:, 0]
        coverage_thin_outcomes.extend(coverage_to.tolist())
        width_thin_outcomes.extend(widths_to.tolist())

    # Store results
    coverage_results[n] = {
        "classic": np.mean(coverage_classic),
        "thin_outcomes": np.mean(coverage_thin_outcomes),
        "sample_split": np.mean(coverage_sample_split),
    }

    width_results[n] = {
        "classic": np.median(width_classic),
        "thin_outcomes": np.median(width_thin_outcomes),
        "sample_split": np.median(width_sample_split),
    }
# %%

