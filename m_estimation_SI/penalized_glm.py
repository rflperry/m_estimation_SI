"""Penalized generalized linear model with robust variance estimation."""

import numpy as np
import regreg.api as rr
from scipy.stats import norm
from scipy.special import expit
from .losses import logistic_loss_smooth, least_squares_loss_smooth


class GLM:
    """Penalized GLM with HC1/CR1 sandwich standard errors.

    Fits a generalized linear model with an optional lasso (L1) penalty
    and an optional affine (e.g. selective-inference) penalty term, via
    the regreg optimization framework.  After fitting, provides HC1
    heteroskedasticity-robust or CR1 cluster-robust Wald confidence
    intervals.

    All losses are mean-scaled (divided by *n*), so penalty parameters
    are comparable across sample sizes.

    Parameters
    ----------
    family : {'linear', 'logistic'}
        Response distribution.  ``'linear'`` uses least-squares loss;
        ``'logistic'`` uses binary cross-entropy loss.
    l1_penalty : float or None, default None
        L1 (lasso) penalty weight ``lambda >= 0``.  ``None`` or ``0``
        fits an unpenalized model.
    affine_penalty : ndarray of shape (n_features_with_intercept,) or None
        Optional additive linear term in the objective, passed as the
        *linear_term* of a :class:`regreg.identity_quadratic`.  Used in
        randomized selective-inference to encode the score of the
        randomization.
    intercept : bool, default True
        If ``True``, prepend a column of ones to *X* before fitting.
        The intercept is never penalized.
    min_its : int, default 50
        Minimum number of iterations for the regreg solver.
    tol : float, default 1e-8
        Convergence tolerance for the regreg solver.
    weights : array-like of shape (n_features,) or None, default None
        Per-feature L1 penalty weights.  Length must equal the number of
        columns in *X* (excluding any intercept).  ``None`` gives uniform
        weight 1 to all features.

    Attributes
    ----------
    beta_ : ndarray of shape (n_features_with_intercept,)
        Fitted coefficient vector (including the intercept when
        ``intercept=True``).
    residuals_ : ndarray of shape (n_samples,)
        Response residuals ``y - predict(X)`` on the training data.
    loss_ : smooth_atom
        The regreg loss object (set after :meth:`fit`).

    Examples
    --------
    Fit a penalized logistic regression and inspect the active set:

    >>> import numpy as np
    >>> from m_estimation_SI import GLM
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> beta_true = np.zeros(21); beta_true[1:4] = [1, -1, 1]
    >>> p = 1 / (1 + np.exp(-(np.c_[np.ones(200), X] @ beta_true)))
    >>> y = rng.binomial(1, p).astype(float)
    >>> glm = GLM(family="logistic", l1_penalty=0.05).fit(X, y)
    >>> glm.active()           # indices of selected features (0-indexed, excl. intercept)
    array([...])
    >>> glm.conf_int(X)        # 95 % Wald confidence intervals for all coefficients
    array([...])

    Notes
    -----
    The sandwich covariance matrix is estimated as

    .. math::

        \\widehat{\\text{Cov}}(\\hat{\\beta}) =
            \\frac{1}{n} H^{-1} \\hat{B} H^{-1}

    where :math:`H` is the mean-scaled Hessian of the loss at
    :math:`\\hat{\\beta}` and :math:`\\hat{B}` is the HC1 or CR1 meat
    matrix.  See :meth:`se` for the exact formula.

    References
    ----------
    .. [1] MacKinnon, J. G. & White, H. (1985).  Some heteroskedasticity-
       consistent covariance matrix estimators with improved finite sample
       properties.  *Journal of Econometrics*, 29, 305–325.
    .. [2] Cameron, A. C. & Miller, D. L. (2015).  A practitioner's guide
       to cluster-robust inference.  *Journal of Human Resources*, 50,
       317–372.
    """

    def __init__(
        self,
        family: str,
        l1_penalty=None,
        affine_penalty=None,
        intercept: bool = True,
        min_its: int = 50,
        tol: float = 1e-8,
        weights=None,
    ):
        self.family = family
        self.l1_penalty = l1_penalty
        self.affine_penalty = affine_penalty
        self.min_its = min_its
        self.tol = tol
        self.intercept = intercept
        self.weights = weights

        self.beta_ = None
        self.loss_ = None
        self.residuals_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GLM":
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix.  Do *not* include an intercept column;
            one is added automatically when ``intercept=True``.
        y : ndarray of shape (n_samples,)
            Response vector.  Binary ``{0, 1}`` for ``family='logistic'``;
            continuous for ``family='linear'``.

        Returns
        -------
        self : GLM
            Fitted estimator.
        """
        n = X.shape[0]
        X_fit = np.hstack([np.ones((n, 1)), X]) if self.intercept else X
        self._solve_problem(X_fit, y)
        self.residuals_ = y - self.predict(X)
        return self

    def _solve_problem(self, X: np.ndarray, y: np.ndarray):
        """Build and solve the regreg problem on the (possibly augmented) design."""
        if self.family == "logistic":
            self.loss_ = logistic_loss_smooth(X, y)
        elif self.family == "linear":
            self.loss_ = least_squares_loss_smooth(X, y)
        else:
            raise ValueError(
                f"family='{self.family}' is not supported. "
                "Choose 'linear' or 'logistic'."
            )

        if self.weights is None:
            w = [0] + [1] * (X.shape[1] - 1) if self.intercept else [1] * X.shape[1]
        else:
            w = [0] + list(self.weights) if self.intercept else list(self.weights)

        lagrange = self.l1_penalty if self.l1_penalty is not None else 0.0
        assert lagrange >= 0, "l1_penalty must be non-negative"
        penalty = rr.weighted_l1norm(w, lagrange=lagrange)

        quad = (
            rr.identity_quadratic(0, 0, self.affine_penalty, 0)
            if self.affine_penalty is not None
            else rr.identity_quadratic(0, 0, 0, 0)
        )

        problem = rr.simple_problem(self.loss_, penalty)
        self.beta_ = problem.solve(quad, min_its=self.min_its, tol=self.tol)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses for new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix (without intercept column).

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted probabilities (logistic) or fitted values (linear).
        """
        assert self.loss_ is not None, "Call fit() before predict()."
        X_pred = np.hstack([np.ones((X.shape[0], 1)), X]) if self.intercept else X
        return self.loss_.predict_(X_pred, self.beta_)

    def coef(self) -> np.ndarray:
        """Fitted coefficient vector including the intercept.

        Returns
        -------
        ndarray of shape (n_features_with_intercept,)
        """
        return self.beta_

    def active(self) -> np.ndarray:
        """Indices of features with non-zero fitted coefficients.

        Returns
        -------
        ndarray of int
            Zero-indexed feature indices relative to the *original* design
            matrix *X* (i.e. the intercept is never included).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.beta_ is None:
            raise ValueError("Call fit() before active().")
        coef = self.beta_[1:] if self.intercept else self.beta_
        return np.where(coef != 0)[0]

    # ------------------------------------------------------------------
    # Variance / standard errors
    # ------------------------------------------------------------------

    def get_var(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        error_model=None,
        clusters=None,
    ):
        """Estimate outcome variances under a working error model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix (without intercept column).
        Y : ndarray of shape (n_samples,)
            Observed responses.
        error_model : {'homogeneous', 'heterogeneous', 'clustered'} or None
            Variance structure to assume.

            * ``None`` or ``'heterogeneous'``: per-observation variances
              (squared residuals for linear; ``mu(1-mu)`` for logistic).
            * ``'homogeneous'``: pooled variance, constant across observations.
            * ``'clustered'``: equicorrelation working covariance shared
              across clusters; returns a ``(cluster_size, cluster_size)``
              matrix rather than a per-observation vector.
        clusters : ndarray of shape (n_samples,) or None
            Integer cluster labels.  Required when
            ``error_model='clustered'``.

        Returns
        -------
        var : ndarray
            * shape ``(n_samples,)`` for ``None``, ``'heterogeneous'``,
              and ``'homogeneous'``.
            * shape ``(cluster_size, cluster_size)`` for ``'clustered'``.

        Raises
        ------
        ValueError
            If *error_model* is not recognised.
        """
        assert self.loss_ is not None, "Call fit() before get_var()."
        X_fit = np.hstack([np.ones((X.shape[0], 1)), X]) if self.intercept else X
        n, p = X_fit.shape

        var_est = self.loss_.get_var_(X_fit, self.beta_, Y)

        if error_model == "homogeneous":
            pdim = np.sum(self.beta_ != 0)
            return np.ones(n) * np.sum(var_est) / (n - pdim)

        if error_model is None or error_model == "heterogeneous":
            return var_est

        if error_model == "clustered" and clusters is not None:
            unique_clusters = np.unique(clusters)
            G = len(unique_clusters)
            q = n // G
            B = np.zeros((q, q))
            for g in unique_clusters:
                idx = np.where(clusters == g)[0]
                Rg = np.sqrt(var_est[idx])
                cluster_cov = np.outer(Rg, Rg)
                diag_mean = np.mean(np.diag(cluster_cov))
                off_diag_mean = np.mean(
                    cluster_cov[np.triu_indices_from(cluster_cov, k=1)]
                )
                B += off_diag_mean * np.ones((len(idx), len(idx))) + (
                    diag_mean - off_diag_mean
                ) * np.eye(len(idx))
            return B / (G - 1)

        raise ValueError(
            f"error_model='{error_model}' is not recognised. "
            "Choose 'homogeneous', 'heterogeneous', or 'clustered'."
        )

    def se(
        self,
        X: np.ndarray,
        cov=False,
        clusters=None,
        correction=None,
    ) -> np.ndarray:
        """Sandwich estimate of the asymptotic standard deviation.

        Returns the square root of the diagonal of
        :math:`H^{-1} \\hat{B} H^{-1}`, where :math:`H` is the
        mean-scaled Hessian and :math:`\\hat{B}` is the HC1 (no clusters)
        or CR1 (clustered) meat matrix.

        .. note::

            This returns the **asymptotic standard deviation** (root of
            :math:`V` where :math:`\\sqrt{n}(\\hat{\\beta} - \\beta)
            \\to N(0, V)`), *not* the standard error of
            :math:`\\hat{\\beta}`.  Divide by :math:`\\sqrt{n}` to
            obtain the standard error, or use :meth:`conf_int` directly.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix used for fitting (without intercept column).
        cov : bool, default False
            If true, returns the full covariance matrix instead.
        clusters : ndarray of shape (n_samples,) or None
            Integer cluster labels.  If provided, CR1 cluster-robust
            standard errors are computed; otherwise HC1 is used.
        correction : ndarray of shape (n_features_with_intercept,
                                       n_features_with_intercept) or None
            Optional additive correction to the meat matrix :math:`\\hat{B}`.
            Used by selective-inference procedures to incorporate
            randomization variance.

        Returns
        -------
        ndarray of shape (n_features_with_intercept,)
            Asymptotic standard deviations (not standard errors).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.beta_ is None:
            raise ValueError("Call fit() before se().")

        X_fit = np.hstack([np.ones((X.shape[0], 1)), X]) if self.intercept else X
        n, p = X_fit.shape
        H_inv = np.linalg.inv(self.loss_.get_hessian(X_fit, self.beta_))

        if clusters is None:
            # HC1 heteroskedasticity-robust meat
            Sigma = X_fit.T @ np.diag(self.residuals_**2) @ X_fit * (n - 1) / (n - p) / n
        else:
            # CR1 cluster-robust meat
            unique_clusters = np.unique(clusters)
            G = len(unique_clusters)
            Sigma = np.zeros((p, p))
            for g in unique_clusters:
                idx = np.where(clusters == g)[0]
                Xg, Rg = X_fit[idx], self.residuals_[idx]
                Sigma += Xg.T @ np.outer(Rg, Rg) @ Xg
            Sigma *= (G / (G - 1)) * ((n - 1) / (n - p)) / n

        if correction is not None:
            Sigma = Sigma + correction

        if cov:
            return H_inv @ Sigma @ H_inv
        return np.sqrt(np.diag(H_inv @ Sigma @ H_inv))

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def conf_int(
        self,
        X: np.ndarray,
        level: float = 0.95,
        clusters=None,
        correction=None,
    ) -> np.ndarray:
        """Wald confidence intervals for all coefficients.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix used for fitting (without intercept column).
        level : float, default 0.95
            Nominal coverage level, e.g. ``0.90`` for 90 % intervals.
        clusters : ndarray of shape (n_samples,) or None
            Integer cluster labels.  If provided, CR1 cluster-robust
            standard errors are used; otherwise HC1.
        correction : ndarray of shape (n_features_with_intercept,
                                       n_features_with_intercept) or None
            Additive correction to the sandwich meat; see :meth:`se`.

        Returns
        -------
        ndarray of shape (n_features_with_intercept, 2)
            Columns are ``[lower, upper]``.  Row 0 corresponds to the
            intercept when ``intercept=True``.
        """
        se = self.se(X, clusters=clusters, correction=correction) / np.sqrt(X.shape[0])
        q = norm.ppf(1 - (1 - level) / 2)
        return np.column_stack([self.beta_ - q * se, self.beta_ + q * se])
