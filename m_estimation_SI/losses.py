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
            \\left[ \\log(1 + e^{x_i^\\top \\beta}) - y_i x_i^\\top \\beta \\right],

    i.e. the mean negative Bernoulli log-likelihood.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.  Include an intercept column explicitly if desired.
    y : ndarray of shape (n_samples,)
        Binary response with values in ``{0, 1}``.

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

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        n, p = X.shape
        if sum(y) > n or sum(y) < 0:
            raise ValueError(
                f"sum(y) = {sum(y):.4g} is outside [0, {n}]; "
                "the logistic log-likelihood is not convex."
            )
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
        Xbeta = self.X @ beta
        f = np.mean(np.logaddexp(0, Xbeta) - self.y * Xbeta)

        if mode == "func":
            return f

        g = self.X.T @ (expit(Xbeta) - self.y) / self.X.shape[0]

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
            ``mu_i = sigmoid(x_i^\\top beta)``.
        """
        mu = self.predict_(X, beta)
        return (X * (mu * (1 - mu))[:, None]).T @ X / X.shape[0]

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
        mu = self.predict_(X, beta)
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

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        _, p = X.shape
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
        residuals = self.y - self.X @ beta
        f = 0.5 * np.mean(residuals**2)

        if mode == "func":
            return f

        g = -self.X.T @ residuals / n

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
        return X.T @ X / X.shape[0]

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
            ``(y_i - x_i^\\top \\beta)^2`` for each observation.
        """
        return (y - X @ beta) ** 2


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

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        _, p = X.shape
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
        Xbeta = self.X @ beta
        mu = np.exp(Xbeta)
        f = np.mean(mu - self.y * Xbeta)

        if mode == "func":
            return f

        g = self.X.T @ (mu - self.y) / self.X.shape[0]

        if mode == "grad":
            return g
        if mode == "both":
            return f, g
        raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Predicted Poisson means.

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
            ``mu_i = exp(x_i^\\top beta)``.
        """
        mu = self.predict_(X, beta)
        return (X * mu[:, None]).T @ X / X.shape[0]

    def get_var_(self, X: np.ndarray, beta: np.ndarray, y=None) -> np.ndarray:
        """Per-observation variance under the Poisson model.

        For Poisson, variance equals the mean.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        beta : ndarray of shape (n_features,)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples,)
            ``exp(x_i^\\top beta)`` for each observation.
        """
        return self.predict_(X, beta)
