"""Loss functions for penalized GLMs.

Each loss is a :class:`regreg.smooth.smooth_atom` subclass, making it
composable with regreg penalties (e.g. ``weighted_l1norm``) via
:class:`regreg.problems.simple_problem`.  All losses are mean-scaled
(divided by *n*), which keeps gradients and penalty parameters on a
consistent scale regardless of sample size.
"""

import numpy as np
from regreg.smooth import smooth_atom
from scipy.special import expit



class logistic_loss_smooth(smooth_atom):
    """Mean-scaled negative log-likelihood for binary logistic regression.

    The loss is

    .. math::

        \\ell(\\beta) = \\frac{1}{n} \\sum_{i=1}^n
            w_i \\left[ \\log(1 + e^{x_i^\\top \\beta}) - y_i x_i^\\top \\beta \\right],

    i.e. the weighted mean negative Bernoulli log-likelihood.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.  Include an intercept column explicitly if desired.
    y : ndarray of shape (n_samples,)
        Binary response with values in ``{0, 1}``.
    obs_weights : ndarray of shape (n_samples,) or None
        Per-observation loss weights ``w_i``.  ``None`` defaults to all ones.

    Raises
    ------
    ValueError
        If ``sum(y)`` lies outside ``[0, n_samples]``.

    Notes
    -----
    Dividing by *n* ensures that a fixed penalty parameter ``lambda``
    has the same effect at different sample sizes, matching the convention
    used by :class:`~m_estimation_SI.GLM`.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, offset=None, obs_weights=None):
        self.X = X
        self.y = y
        n, p = X.shape
        if sum(y) > n or sum(y) < 0:
            raise ValueError(
                f"sum(y) = {sum(y):.4g} is outside [0, {n}]; "
                "the logistic log-likelihood is not convex."
            )
        self._glm_offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.obs_weights = np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)
        super().__init__(shape=(p,))

    def smooth_objective(
        self,
        beta: np.ndarray,
        mode: str = "both",
        check_feasibility=None,
    ):
        """Evaluate the loss and/or its gradient.

        Parameters
        ----------
        beta : ndarray of shape (n_features,)
            Coefficient vector at which to evaluate.
        mode : {'func', 'grad', 'both'}
            What to compute.  ``'func'`` returns the scalar loss;
            ``'grad'`` returns the gradient; ``'both'`` returns both.

        Returns
        -------
        f : float
            Loss value.  Present when *mode* is ``'func'`` or ``'both'``.
        g : ndarray of shape (n_features,)
            Gradient.  Present when *mode* is ``'grad'`` or ``'both'``.

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted strings.
        """
        eta = self.X @ beta + self._glm_offset
        f = np.mean(self.obs_weights * (np.logaddexp(0, eta) - self.y * eta))

        if mode == "func":
            return f

        g = self.X.T @ (self.obs_weights * (expit(eta) - self.y)) / self.X.shape[0]

        if mode == "grad":
            return g
        if mode == "both":
            return f, g
        raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Predicted probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted probabilities in ``(0, 1)``.
        """
        return expit(X @ beta)

    def _fitted_values(self, beta: np.ndarray) -> np.ndarray:
        """Fitted probabilities at training data including any offset."""
        return expit(self.X @ beta + self._glm_offset)

    def get_hessian(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Mean-scaled Fisher information matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_features, n_features)
            ``(1/n) X^\\top W X`` where
            ``W = diag(mu_i (1 - mu_i))`` and
            ``mu_i = sigmoid(x_i^\\top beta + offset_i)``.
        """
        mu = expit(X @ beta + self._glm_offset)
        return (X * (self.obs_weights * mu * (1 - mu))[:, None]).T @ X / X.shape[0]

    def get_var_(self, X: np.ndarray, beta: np.ndarray, y=None) -> np.ndarray:
        """Per-observation variance under the logistic model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples,)
            ``mu_i (1 - mu_i)`` evaluated at the fitted probabilities.
        """
        mu = expit(X @ beta + self._glm_offset)
        return mu * (1 - mu)

    def get_model_var_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """GLM variance function V(mu) = mu(1-mu) for the Bernoulli family."""
        mu = expit(X @ beta + self._glm_offset)
        return mu * (1 - mu)


class least_squares_loss_smooth(smooth_atom):
    """Mean-scaled least-squares loss for linear regression.

    The loss is

    .. math::

        \\ell(\\beta) = \\frac{1}{2n} \\|y - X\\beta\\|^2.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.  Include an intercept column explicitly if desired.
    y : ndarray of shape (n_samples,)
        Continuous response vector.

    Notes
    -----
    Dividing by *n* ensures that a fixed penalty parameter ``lambda``
    has the same effect at different sample sizes, matching the convention
    used by :class:`~m_estimation_SI.GLM`.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, offset=None, obs_weights=None):
        self.X = X
        self.y = y
        n, p = X.shape
        self._glm_offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.obs_weights = np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)
        super().__init__(shape=(p,))

    def smooth_objective(
        self,
        beta: np.ndarray,
        mode: str = "both",
        check_feasibility=None,
    ):
        """Evaluate the loss and/or its gradient.

        Parameters
        ----------
        beta : ndarray of shape (n_features,)
            Coefficient vector at which to evaluate.
        mode : {'func', 'grad', 'both'}
            What to compute.  ``'func'`` returns the scalar loss;
            ``'grad'`` returns the gradient; ``'both'`` returns both.

        Returns
        -------
        f : float
            Loss value.  Present when *mode* is ``'func'`` or ``'both'``.
        g : ndarray of shape (n_features,)
            Gradient.  Present when *mode* is ``'grad'`` or ``'both'``.

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted strings.
        """
        n = self.X.shape[0]
        residuals = self.y - (self.X @ beta + self._glm_offset)
        f = 0.5 * np.mean(self.obs_weights * residuals**2)

        if mode == "func":
            return f

        g = -self.X.T @ (self.obs_weights * residuals) / n

        if mode == "grad":
            return g
        if mode == "both":
            return f, g
        raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Predicted values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return X @ beta

    def get_hessian(self, X: np.ndarray, beta=None) -> np.ndarray:
        """Mean-scaled Hessian.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ignored

        Returns
        -------
        ndarray of shape (n_features, n_features)
            ``(1/n) X^\\top X``.
        """
        return (X * self.obs_weights[:, None]).T @ X / X.shape[0]

    def get_var_(self, X: np.ndarray, beta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-observation squared residuals.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)
        y : ndarray of shape (n_samples,)
            Observed response.

        Returns
        -------
        ndarray of shape (n_samples,)
            ``(y_i - x_i^\\top \\beta - offset_i)^2`` for each observation.
        """
        return (y - X @ beta - self._glm_offset) ** 2

    def _fitted_values(self, beta: np.ndarray) -> np.ndarray:
        """Fitted values at training data including any offset."""
        return self.X @ beta + self._glm_offset

    def get_model_var_(self, X: np.ndarray, _beta: np.ndarray) -> np.ndarray:
        """GLM variance function V(mu) = 1 for the Gaussian family."""
        return np.ones(X.shape[0])


class poisson_loss_smooth(smooth_atom):
    """Mean-scaled negative log-likelihood for Poisson regression.

    The loss is

    .. math::

        \\ell(\\beta) = \\frac{1}{n} \\sum_{i=1}^n
            \\left[ e^{x_i^\\top \\beta} - y_i x_i^\\top \\beta \\right],

    i.e. the mean negative Poisson log-likelihood (constant ``log(y_i!)``
    terms are dropped as they do not affect optimisation).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.  Include an intercept column explicitly if desired.
    y : ndarray of shape (n_samples,)
        Response vector.  Typically non-negative integer counts, but any
        real-valued ``y`` is accepted (e.g. thinned pseudo-outcomes).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, offset=None, obs_weights=None):
        self.X = X
        self.y = y
        n, p = X.shape
        self._glm_offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.obs_weights = np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)
        super().__init__(shape=(p,))

    def smooth_objective(
        self,
        beta: np.ndarray,
        mode: str = "both",
        check_feasibility=None,
    ):
        """Evaluate the loss and/or its gradient.

        Parameters
        ----------
        beta : ndarray of shape (n_features,)
        mode : {'func', 'grad', 'both'}

        Returns
        -------
        f : float
            Loss value.  Present when *mode* is ``'func'`` or ``'both'``.
        g : ndarray of shape (n_features,)
            Gradient.  Present when *mode* is ``'grad'`` or ``'both'``.

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted strings.
        """
        eta = self.X @ beta + self._glm_offset
        mu = np.exp(eta)
        f = np.mean(self.obs_weights * (mu - self.y * eta))

        if mode == "func":
            return f

        g = self.X.T @ (self.obs_weights * (mu - self.y)) / self.X.shape[0]

        if mode == "grad":
            return g
        if mode == "both":
            return f, g
        raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Predicted Poisson means (no offset; for new data).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted means ``exp(X @ beta)``, all positive.
        """
        return np.exp(X @ beta)

    def _fitted_values(self, beta: np.ndarray) -> np.ndarray:
        """Fitted means at training data including any offset."""
        return np.exp(self.X @ beta + self._glm_offset)

    def get_hessian(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Mean-scaled Fisher information matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_features, n_features)
            ``(1/n) X^\\top \\text{diag}(\\mu) X`` where
            ``mu_i = exp(x_i^\\top beta + offset_i)``.
        """
        mu = np.exp(X @ beta + self._glm_offset)
        return (X * (self.obs_weights * mu)[:, None]).T @ X / X.shape[0]

    def get_var_(self, X: np.ndarray, beta: np.ndarray, _y=None) -> np.ndarray:
        """Per-observation Poisson variance (equals the fitted mean).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)
        _y : ignored

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return np.exp(X @ beta + self._glm_offset)

    def get_model_var_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """GLM variance function V(mu) = mu for the Poisson family."""
        return np.exp(X @ beta + self._glm_offset)


class negative_binomial_loss_smooth(smooth_atom):
    """Mean-scaled negative log-likelihood for negative binomial regression.

    The loss is

    .. math::

        \\ell(\\beta) = \\frac{1}{n} \\sum_{i=1}^n
            \\left[ (\\theta + y_i)\\log(\\theta + e^{x_i^\\top \\beta})
                    - y_i\\, x_i^\\top \\beta \\right],

    i.e. the mean negative NB log-likelihood with a log link, dropping
    terms that are constant in :math:`\\beta`.  The dispersion
    :math:`\\theta > 0` is treated as fixed (known).

    As :math:`\\theta \\to \\infty` the NB approaches Poisson; small
    :math:`\\theta` corresponds to high overdispersion.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.  Include an intercept column explicitly if desired.
    y : ndarray of shape (n_samples,)
        Response vector.  Any real value is accepted; negative or continuous
        values arise naturally in outcome-thinning procedures.
    theta : float
        Positive dispersion (size) parameter.

    Raises
    ------
    ValueError
        If ``theta <= 0``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, theta: float, offset=None, obs_weights=None):
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.X = X
        self.y = y
        self.theta = float(theta)
        n, p = X.shape
        self._glm_offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        self.obs_weights = np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)
        super().__init__(shape=(p,))

    def smooth_objective(
        self,
        beta: np.ndarray,
        mode: str = "both",
        check_feasibility=None,
    ):
        """Evaluate the loss and/or its gradient.

        Parameters
        ----------
        beta : ndarray of shape (n_features,)
        mode : {'func', 'grad', 'both'}

        Returns
        -------
        f : float
            Loss value.  Present when *mode* is ``'func'`` or ``'both'``.
        g : ndarray of shape (n_features,)
            Gradient.  Present when *mode* is ``'grad'`` or ``'both'``.

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted strings.
        """
        theta = self.theta
        eta = self.X @ beta + self._glm_offset
        mu = np.exp(eta)
        f = np.mean(self.obs_weights * ((theta + self.y) * np.log(theta + mu) - self.y * eta))

        if mode == "func":
            return f

        # gradient: (theta / n) * X^T [(mu - y) / (theta + mu)]
        g = self.X.T @ (self.obs_weights * theta * (mu - self.y) / (theta + mu)) / self.X.shape[0]

        if mode == "grad":
            return g
        if mode == "both":
            return f, g
        raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Predicted NB means (no offset; for new data).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted means, all positive.
        """
        return np.exp(X @ beta)

    def _fitted_values(self, beta: np.ndarray) -> np.ndarray:
        """Fitted means at training data including any offset."""
        return np.exp(self.X @ beta + self._glm_offset)

    def get_hessian(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Mean-scaled Fisher information matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)

        Returns
        -------
        ndarray of shape (n_features, n_features)
            ``(1/n) X^\\top \\mathrm{diag}(w) X`` where
            ``w_i = theta * mu_i / (theta + mu_i)`` and
            ``mu_i = exp(x_i^\\top beta + offset_i)``.
        """
        mu = np.exp(X @ beta + self._glm_offset)
        w = self.obs_weights * self.theta * mu / (self.theta + mu)
        return (X * w[:, None]).T @ X / X.shape[0]

    def get_var_(self, X: np.ndarray, beta: np.ndarray, _y=None) -> np.ndarray:
        """Per-observation NB variance: ``mu + mu^2 / theta``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)
        _y : ignored

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        mu = np.exp(X @ beta + self._glm_offset)
        return mu + mu ** 2 / self.theta

    def get_model_var_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """GLM variance function V(mu) = mu + mu^2/theta for the NB family."""
        mu = np.exp(X @ beta + self._glm_offset)
        return mu + mu ** 2 / self.theta
