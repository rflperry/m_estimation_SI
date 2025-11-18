import numpy as np
import regreg.api as rr
from scipy.stats import norm
from scipy.special import expit
from .losses import logistic_loss_smooth, least_squares_loss_smooth

class GLM:
    """
    GLM using regreg with optional L1 penalty and optional affine penalty.
    """

    def __init__(self, family, l1_penalty=None, affine_penalty=None, gamma=1, intercept=True, min_its=20, tol=1.e-10):
        """
        Parameters
        ----------
        l1_penalty : float
            L1 penalty weight (lambda). 0 means no L1 regularization.
        affine_penalty : regreg proximal_atom, optional
            Any additional convex penalty (e.g., quadratic, group lasso).
        """
        self.family = family
        self.l1_penalty = l1_penalty
        self.affine_penalty = affine_penalty
        self.gamma = gamma
        self.min_its = min_its
        self.tol = tol
        self.intercept = intercept

        self.beta_ = None
        self.problem_ = None
        self.loss_ = None
        self.residuals_ = None

    def fit(self, X, y):
        """
        X : array-like, shape (n_samples, n_features)
            Design matrix
        y : array-like, shape (n_samples,)
        """
        n = X.shape[0]
        if self.intercept:
            self._solve_problem(np.hstack([np.ones((n, 1)), X]), y)
        else:
            self._solve_problem(X, y)

        self.residuals_ = y - self.predict(X)

        return self

    def _solve_problem(self, X, y):
        """Construct regreg loss + penalties."""

        if self.family == "logistic":
            self.loss_ = logistic_loss_smooth(X, y)
        elif self.family == "linear":
            self.loss_ = least_squares_loss_smooth(X, y)
        else:
            raise ValueError("Unsupported family")

        if self.l1_penalty is not None:
            assert self.l1_penalty >= 0, "l1_penalty must be non-negative"
            if self.intercept:
                lam = rr.weighted_l1norm([0] + [1]*(X.shape[1]-1), lagrange=self.l1_penalty)  # to not penalize intercept
            else:
                lam = rr.weighted_l1norm([1]*X.shape[1], lagrange=self.l1_penalty)
        else:
            lam = rr.weighted_l1norm([1]*X.shape[1], lagrange=0)  # no penalty

        if self.affine_penalty is not None:
            quad = rr.identity_quadratic(0, 0, self.affine_penalty, 0)
        else:
            quad = rr.identity_quadratic(0, 0, 0, 0)  # no penalty

        problem = rr.simple_problem(self.loss_, lam)

        self.beta_ = problem.solve(quad, min_its=self.min_its, tol=self.tol)

    def predict(self, X):
        """Predicts new outcomes"""
        assert self.loss_ is not None, "Model must be fit before it can predict."
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.loss_.predict_(X, self.beta_)

    def coef(self):
        """Return estimated coefficients including intercept."""
        return self.beta_
    
    def get_var(self, X, Y, error_model=None):
        """Estimates the outcome variances using the working model."""
        assert self.loss_ is not None, "Model must be fit before it can predict."
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        var_est = self.loss_.get_var_(X, self.beta_, Y)
        if error_model == "homogeneous":
            n, p = X.shape
            return np.ones_like(var_est) * (np.mean(var_est) / (n - p))
        elif error_model == "heterogeneous" or error_model is None:
            return var_est
        else:
            raise ValueError(f"error_model {error_model} not recognized.")

    def active(self):
        """Return indices of nonzero coefficients (excluding intercept)."""
        if self.beta_ is None:
            raise ValueError("Model not fitted yet")

        if self.intercept:
            return np.where(self.beta_[1:] != 0)[0]
        else:
            return np.where(self.beta_ != 0)[0]

    def se(self, X, Y_var=None):
        """Compute approximate standard errors using sandwich formula."""
        if self.beta_ is None:
            raise ValueError("Model not fitted yet")

        if Y_var is not None and isinstance(Y_var, (float)):
            W = np.diag(np.ones(X.shape[0]) * Y_var)
        elif Y_var is not None:
            assert len(Y_var) == X.shape[0]
            W = np.diag(Y_var)
        else:
            W = np.diag(self.residuals_ ** 2)  # sandwich model estimate

        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        Sigma = X.T @ W @ X / X.shape[0]
        H_est_inv = np.linalg.inv(self.loss_.get_hessian(X, self.beta_))
        cov_beta_hat = H_est_inv @ Sigma @ H_est_inv
        se = np.sqrt(np.diag(cov_beta_hat))

        return se

    def conf_int(self, X, Y_var=None, level=0.95, var_scale=1.0):
        se = self.se(X, Y_var) * np.sqrt(var_scale) / np.sqrt(X.shape[0])
        q = norm.ppf(1 - (1 - level) / 2, loc=0, scale=1)
        conf_int = np.vstack([self.beta_ - q * se, self.beta_ + q * se]).T  
        return conf_int
