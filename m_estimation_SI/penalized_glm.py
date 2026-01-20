import numpy as np
import regreg.api as rr
from scipy.stats import norm
from scipy.special import expit
from .losses import logistic_loss_smooth, least_squares_loss_smooth


class GLM:
    """
    GLM using regreg with optional L1 penalty and optional affine penalty.
    """

    def __init__(
        self,
        family,
        l1_penalty=None,
        affine_penalty=None,
        intercept=True,
        min_its=50,
        tol=1.0e-8,
        weights=None,
    ):
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
        self.min_its = min_its
        self.tol = tol
        self.intercept = intercept
        self.weights = weights

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

        weights = self.weights
        if weights is None:
            if self.intercept:
                weights = [0] + [1] * (X.shape[1] - 1)
            else:
                weights = [1] * X.shape[1]
        elif self.intercept:
            weights = [0] + weights  # no penalty on intercept

        if self.l1_penalty is not None:
            assert self.l1_penalty >= 0, "l1_penalty must be non-negative"
            lam = rr.weighted_l1norm(weights, lagrange=self.l1_penalty)                
        else:
            lam = rr.weighted_l1norm(weights, lagrange=0)  # no penalty

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

    def get_var(self, X, Y, error_model=None, clusters=None):
        """Estimates the outcome variances using the working model."""
        assert self.loss_ is not None, "Model must be fit before it can predict."
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n, p = X.shape

        var_est = self.loss_.get_var_(X, self.beta_, Y)
        if error_model == "homogeneous":
            pdim = np.sum(self.beta_ != 0)
            return np.ones(n) * np.sum(var_est) / (n - pdim)
        elif error_model == "heterogeneous" or error_model is None:
            return var_est
        elif error_model == "clustered" and clusters is not None:
            unique_clusters = np.unique(clusters)
            G = len(unique_clusters)
            q = n // G  # assuming all are of the same length
            # U = np.zeros((G, q))
            # for i, g in enumerate(unique_clusters):
            #     idx = np.where(clusters == g)[0]
            #     Rg = np.sqrt(var_est[idx])
            #     U[i, :] = Rg
            # Uc = U - U.mean(axis=0)
            # return (Uc.T @ Uc) / (G - 1)
            B = np.zeros((q, q))
            for g in unique_clusters:
                ## Totally free covariance, shared across clusters
                # idx = np.where(clusters == g)[0]
                # Rg = np.sqrt(var_est[idx])
                # B += np.outer(Rg, Rg)
                ## Equicorrelation working covariance
                idx = np.where(clusters == g)[0]
                Rg = np.sqrt(var_est[idx])
                cluster_cov = np.outer(Rg, Rg)
                # Equicorrelation structure: diagonal = mean variance, off-diagonal = mean covariance
                diag_mean = np.mean(np.diag(cluster_cov))
                off_diag_mean = np.mean(cluster_cov[np.triu_indices_from(cluster_cov, k=1)])
                cluster_cov_eq = off_diag_mean * np.ones((len(idx), len(idx))) + (diag_mean - off_diag_mean) * np.eye(len(idx))
                B += cluster_cov_eq
            return B / (len(unique_clusters) - 1)       
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

    def se(self, X, clusters=None, correction=None):
        """Compute approximate standard errors using sandwich formula."""
        if self.beta_ is None:
            raise ValueError("Model not fitted yet")

        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n, p = X.shape

        H_inv = np.linalg.inv(self.loss_.get_hessian(X, self.beta_))
        # W = np.diag(self.residuals_ ** 2)  # sandwich model estimate
        # Sigma = X.T @ W @ X / (X.shape[0] - X.shape[1])
        # cov_beta_hat = H_est_inv @ Sigma @ H_est_inv
        # se = np.sqrt(np.diag(cov_beta_hat))

        # --- HC1 sandwich standard errors ---
        if clusters is None:
            W = np.diag(self.residuals_**2)
            Sigma = X.T @ W @ X * ((n - 1) / (n - p)) / n
            if correction is not None:
                cov = H_inv @ (Sigma + correction) @ H_inv
            else:
                cov = H_inv @ Sigma @ H_inv
            return np.sqrt(np.diag(cov))

        # --- CR1 cluster-robust standard errors ---
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)

        # Clustered meat matrix
        B = np.zeros((p, p))

        for g in unique_clusters:
            idx = np.where(clusters == g)[0]
            Xg = X[idx, :]
            Rg = self.residuals_[idx]
            B += Xg.T @ np.outer(Rg, Rg) @ Xg

        # CR1 finite-sample correction
        df_correction = (G / (G - 1)) * ((n - 1) / (n - p)) / n
        B *= df_correction

        # Sandwich estimator
        if correction is not None:
            cov = H_inv @ (B + correction) @ H_inv
        else:
            cov = H_inv @ B @ H_inv
        se = np.sqrt(np.diag(cov))

        return se

    def conf_int(self, X, level=0.95, clusters=None, correction=None):
        se = self.se(X, clusters=clusters, correction=correction) / np.sqrt(X.shape[0])
        q = norm.ppf(1 - (1 - level) / 2, loc=0, scale=1)
        conf_int = np.vstack([self.beta_ - q * se, self.beta_ + q * se]).T
        return conf_int
