import numpy as np
import regreg.api as rr
from scipy.stats import norm
from scipy.special import expit
from .losses import logistic_loss_smooth

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

        self.beta_hat_ = None
        self.problem_ = None        

    def fit(self, X, y):
        """
        X : array-like, shape (n_samples, n_features)
            Design matrix
        y : array-like, shape (n_samples,)
        """
        self.n_, self.p_ = X.shape
        if self.intercept:
            X = np.hstack([np.ones((self.n_, 1)), X])

        self._solve_problem(X, y)

        return self

    def _solve_problem(self, X, y):
        """Construct regreg loss + penalties."""
        
        if self.family == "logistic":
            loss = logistic_loss_smooth(X, y)
        else:
            raise ValueError("Unsupported family")

        if self.l1_penalty is not None:
            assert self.l1_penalty >= 0, "l1_penalty must be non-negative"
            if self.intercept:
                lam = rr.weighted_l1norm([0] + [1]*self.p_, lagrange=self.l1_penalty)  # to not penalize intercept
            else:
                lam = rr.weighted_l1norm([1]*self.p_, lagrange=self.l1_penalty)
        else:
            lam = rr.weighted_l1norm([1]*X.shape[1], lagrange=0)  # no penalty

        if self.affine_penalty is not None:
            quad = rr.identity_quadratic(0, 0, self.affine_penalty, 0)
        else:
            quad = rr.identity_quadratic(0, 0, 0, 0)  # no penalty

        problem = rr.simple_problem(loss, lam)

        self.beta_ = problem.solve(quad, min_its=self.min_its, tol=self.tol)

    def coef(self):
        """Return estimated coefficients including intercept."""
        return self.beta_

    def active(self):
        """Return indices of nonzero coefficients (excluding intercept)."""
        if self.beta_ is None:
            raise ValueError("Model not fitted yet")
        elif self.intercept:
            return np.where(self.beta_[1:] != 0)[0]
        else:
            return np.where(self.beta_ != 0)[0]

    def se(self, X, y_var=None):
        """Compute approximate standard errors using sandwich formula."""
        if self.beta_ is None:
            raise ValueError("Model not fitted yet")

        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        eta = X @ self.beta_
        mu = expit(eta)  # to ovoid numerical overflow
        if y_var is not None and isinstance(y_var, (float)):
            W = np.ones(self.n_) * y_var
        elif y_var is not None:
            assert len(y_var) == len(mu)
            W = y_var
        else:
            W = mu * (1 - mu)  # working model variance estimate

        Sigma = (X.T * W) @ X / self.n_
        Y_var_est = mu * (1 - mu)
        H_est_inv = np.linalg.inv((X.T * Y_var_est) @ X / self.n_)
        cov_beta_hat = H_est_inv @ Sigma @ H_est_inv / self.n_
        se = np.sqrt(np.diag(cov_beta_hat))

        return se

    def conf_int(self, X, y_var=None, level=0.95):
        se = self.se(X, y_var)
        q = norm.ppf(1 - (1 - level) / 2, loc=0, scale=1)
        conf_int = np.vstack([self.beta_ - q * se, self.beta_ + q * se]).T  
        return conf_int
