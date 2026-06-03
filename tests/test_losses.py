"""Tests for logistic_loss_smooth and least_squares_loss_smooth."""

import numpy as np
import pytest
from m_estimation_SI.losses import logistic_loss_smooth, least_squares_loss_smooth, poisson_loss_smooth

RNG = np.random.default_rng(0)
N, P = 80, 5
X = RNG.standard_normal((N, P))
BETA = RNG.standard_normal(P)
Y_BIN = (RNG.standard_normal(N) > 0).astype(float)
Y_CONT = RNG.standard_normal(N)
Y_COUNT = RNG.poisson(3, size=N).astype(float)  # Poisson counts


# ---------------------------------------------------------------------------
# logistic_loss_smooth
# ---------------------------------------------------------------------------

class TestLogisticLoss:
    def setup_method(self):
        self.loss = logistic_loss_smooth(X, Y_BIN)

    def test_func_mode_returns_scalar(self):
        f = self.loss.smooth_objective(BETA, mode="func")
        assert np.isscalar(f) or f.ndim == 0
        assert f > 0

    def test_grad_mode_returns_correct_shape(self):
        g = self.loss.smooth_objective(BETA, mode="grad")
        assert g.shape == (P,)

    def test_both_mode_consistent_with_separate(self):
        f_both, g_both = self.loss.smooth_objective(BETA, mode="both")
        f_func = self.loss.smooth_objective(BETA, mode="func")
        g_grad = self.loss.smooth_objective(BETA, mode="grad")
        assert np.isclose(f_both, f_func)
        assert np.allclose(g_both, g_grad)

    def test_gradient_matches_finite_difference(self):
        eps = 1e-5
        _, g = self.loss.smooth_objective(BETA, mode="both")
        g_fd = np.zeros(P)
        for j in range(P):
            beta_plus = BETA.copy(); beta_plus[j] += eps
            beta_minus = BETA.copy(); beta_minus[j] -= eps
            f_plus = self.loss.smooth_objective(beta_plus, mode="func")
            f_minus = self.loss.smooth_objective(beta_minus, mode="func")
            g_fd[j] = (f_plus - f_minus) / (2 * eps)
        assert np.allclose(g, g_fd, atol=1e-5)

    def test_predict_is_probability(self):
        p_hat = self.loss.predict_(X, BETA)
        assert p_hat.shape == (N,)
        assert np.all(p_hat > 0) and np.all(p_hat < 1)

    def test_hessian_shape_and_psd(self):
        H = self.loss.get_hessian(X, BETA)
        assert H.shape == (P, P)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals >= -1e-10), "Hessian must be positive semi-definite"

    def test_var_in_valid_range(self):
        var = self.loss.get_var_(X, BETA)
        assert var.shape == (N,)
        assert np.all(var >= 0) and np.all(var <= 0.25 + 1e-12)

    def test_invalid_y_raises(self):
        y_bad = np.full(N, 2.0)
        with pytest.raises(ValueError):
            logistic_loss_smooth(X, y_bad)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self.loss.smooth_objective(BETA, mode="bad")


# ---------------------------------------------------------------------------
# least_squares_loss_smooth
# ---------------------------------------------------------------------------

class TestLeastSquaresLoss:
    def setup_method(self):
        self.loss = least_squares_loss_smooth(X, Y_CONT)

    def test_func_mode_returns_nonnegative_scalar(self):
        f = self.loss.smooth_objective(BETA, mode="func")
        assert np.isscalar(f) or f.ndim == 0
        assert f >= 0

    def test_grad_mode_returns_correct_shape(self):
        g = self.loss.smooth_objective(BETA, mode="grad")
        assert g.shape == (P,)

    def test_both_mode_consistent_with_separate(self):
        f_both, g_both = self.loss.smooth_objective(BETA, mode="both")
        f_func = self.loss.smooth_objective(BETA, mode="func")
        g_grad = self.loss.smooth_objective(BETA, mode="grad")
        assert np.isclose(f_both, f_func)
        assert np.allclose(g_both, g_grad)

    def test_gradient_matches_finite_difference(self):
        eps = 1e-5
        _, g = self.loss.smooth_objective(BETA, mode="both")
        g_fd = np.zeros(P)
        for j in range(P):
            beta_plus = BETA.copy(); beta_plus[j] += eps
            beta_minus = BETA.copy(); beta_minus[j] -= eps
            f_plus = self.loss.smooth_objective(beta_plus, mode="func")
            f_minus = self.loss.smooth_objective(beta_minus, mode="func")
            g_fd[j] = (f_plus - f_minus) / (2 * eps)
        assert np.allclose(g, g_fd, atol=1e-5)

    def test_func_is_zero_at_exact_fit(self):
        beta_exact = np.linalg.lstsq(X, Y_CONT, rcond=None)[0]
        loss_exact = least_squares_loss_smooth(X, X @ beta_exact)
        f = loss_exact.smooth_objective(beta_exact, mode="func")
        assert np.isclose(f, 0.0, atol=1e-10)

    def test_predict_is_linear(self):
        y_hat = self.loss.predict_(X, BETA)
        assert np.allclose(y_hat, X @ BETA)

    def test_hessian_formula(self):
        H = self.loss.get_hessian(X, BETA)
        assert H.shape == (P, P)
        assert np.allclose(H, X.T @ X / N)

    def test_hessian_is_psd(self):
        H = self.loss.get_hessian(X, BETA)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals >= -1e-10)

    def test_var_equals_squared_residuals(self):
        var = self.loss.get_var_(X, BETA, y=Y_CONT)
        expected = (Y_CONT - X @ BETA) ** 2
        assert np.allclose(var, expected)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self.loss.smooth_objective(BETA, mode="bad")


# ---------------------------------------------------------------------------
# poisson_loss_smooth
# ---------------------------------------------------------------------------

class TestPoissonLoss:
    def setup_method(self):
        # Use small BETA to avoid exp overflow
        self.beta = BETA * 0.1
        self.loss = poisson_loss_smooth(X, Y_COUNT)

    def test_func_mode_returns_nonnegative_scalar(self):
        f = self.loss.smooth_objective(self.beta, mode="func")
        assert np.isscalar(f) or f.ndim == 0
        assert f >= 0

    def test_grad_mode_returns_correct_shape(self):
        g = self.loss.smooth_objective(self.beta, mode="grad")
        assert g.shape == (P,)

    def test_both_mode_consistent_with_separate(self):
        f_both, g_both = self.loss.smooth_objective(self.beta, mode="both")
        f_func = self.loss.smooth_objective(self.beta, mode="func")
        g_grad = self.loss.smooth_objective(self.beta, mode="grad")
        assert np.isclose(f_both, f_func)
        assert np.allclose(g_both, g_grad)

    def test_gradient_matches_finite_difference(self):
        eps = 1e-5
        _, g = self.loss.smooth_objective(self.beta, mode="both")
        g_fd = np.zeros(P)
        for j in range(P):
            beta_plus = self.beta.copy(); beta_plus[j] += eps
            beta_minus = self.beta.copy(); beta_minus[j] -= eps
            f_plus = self.loss.smooth_objective(beta_plus, mode="func")
            f_minus = self.loss.smooth_objective(beta_minus, mode="func")
            g_fd[j] = (f_plus - f_minus) / (2 * eps)
        assert np.allclose(g, g_fd, atol=1e-5)

    def test_predict_is_positive(self):
        mu = self.loss.predict_(X, self.beta)
        assert mu.shape == (N,)
        assert np.all(mu > 0)

    def test_hessian_shape_and_psd(self):
        H = self.loss.get_hessian(X, self.beta)
        assert H.shape == (P, P)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals >= -1e-10), "Hessian must be positive semi-definite"

    def test_hessian_formula(self):
        mu = np.exp(X @ self.beta)
        expected = (X * mu[:, None]).T @ X / N
        assert np.allclose(self.loss.get_hessian(X, self.beta), expected)

    def test_var_equals_fitted_mean(self):
        var = self.loss.get_var_(X, self.beta)
        mu = np.exp(X @ self.beta)
        assert np.allclose(var, mu)

    def test_var_is_positive(self):
        var = self.loss.get_var_(X, self.beta)
        assert np.all(var > 0)

    def test_negative_y_accepted(self):
        # Negative pseudo-outcomes (e.g. from Gaussian thinning) are allowed.
        y_neg = np.full(N, -1.0)
        loss = poisson_loss_smooth(X, y_neg)
        f = loss.smooth_objective(self.beta, mode="func")
        assert np.isfinite(f)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            self.loss.smooth_objective(self.beta, mode="bad")
