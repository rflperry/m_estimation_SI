"""Comprehensive tests for the GLM class."""

import numpy as np
import pytest
from m_estimation_SI import GLM

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N, P = 150, 8
LAM = np.sqrt(2 * np.log(P) / N)

# Continuous design matrix
X = RNG.standard_normal((N, P))
X /= np.linalg.norm(X, axis=0)

# Linear data
BETA_TRUE = np.zeros(P + 1)
BETA_TRUE[1:4] = [2.0, -1.5, 1.0]  # intercept + 3 nonzero features
X_INT = np.hstack([np.ones((N, 1)), X])
Y_LINEAR = X_INT @ BETA_TRUE + RNG.standard_normal(N) * 0.5

# Logistic data
ETA = X_INT @ BETA_TRUE
PROBS = 1 / (1 + np.exp(-ETA))
Y_BIN = RNG.binomial(1, PROBS).astype(float)

# Poisson data (small coefficients to keep means well-behaved)
BETA_POISSON = np.zeros(P + 1)
BETA_POISSON[0] = 0.5   # intercept -> baseline mean ~1.6
BETA_POISSON[1:4] = [0.3, -0.2, 0.2]
MU_POISSON = np.exp(X_INT @ BETA_POISSON)
Y_COUNT = RNG.poisson(MU_POISSON).astype(float)

# Negative binomial data (same linear predictor as Poisson, theta=3)
THETA_NB = 3.0
_lambda_nb = RNG.gamma(shape=THETA_NB, scale=1.0 / THETA_NB, size=N)
Y_NB = RNG.poisson(MU_POISSON * _lambda_nb).astype(float)

# Cluster labels (15 clusters of 10)
N_CLUSTERS = 15
CLUSTER_SIZE = N // N_CLUSTERS
CLUSTERS = np.repeat(np.arange(N_CLUSTERS), CLUSTER_SIZE)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestFit:
    def test_logistic_fit_returns_self(self):
        glm = GLM(family="logistic", l1_penalty=LAM)
        result = glm.fit(X, Y_BIN)
        assert result is glm

    def test_linear_fit_returns_self(self):
        glm = GLM(family="linear", l1_penalty=LAM)
        result = glm.fit(X, Y_LINEAR)
        assert result is glm

    def test_logistic_beta_shape_with_intercept(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        assert glm.beta_.shape == (P + 1,)

    def test_linear_beta_shape_with_intercept(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        assert glm.beta_.shape == (P + 1,)

    def test_beta_shape_without_intercept(self):
        glm = GLM(family="linear", l1_penalty=LAM, intercept=False).fit(X, Y_LINEAR)
        assert glm.beta_.shape == (P,)

    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError):
            GLM(family="gamma", l1_penalty=LAM).fit(X, Y_BIN)

    def test_nb_fit_returns_self_with_theta(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM, theta=THETA_NB)
        assert glm.fit(X, Y_NB) is glm

    def test_nb_fit_returns_self_without_theta(self):
        # theta=None triggers IRLS estimation
        glm = GLM(family="negative_binomial", l1_penalty=LAM)
        assert glm.fit(X, Y_NB) is glm

    def test_nb_beta_shape(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM, theta=THETA_NB).fit(X, Y_NB)
        assert glm.beta_.shape == (P + 1,)

    def test_nb_theta_estimated_is_positive(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert glm.theta_ is not None
        assert glm.theta_ > 0

    def test_nb_theta_estimated_set_on_self(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert glm.theta == glm.theta_

    def test_poisson_fit_returns_self(self):
        glm = GLM(family="poisson", l1_penalty=LAM)
        result = glm.fit(X, Y_COUNT)
        assert result is glm

    def test_poisson_beta_shape_with_intercept(self):
        glm = GLM(family="poisson", l1_penalty=LAM).fit(X, Y_COUNT)
        assert glm.beta_.shape == (P + 1,)

    def test_residuals_computed_after_fit(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        assert glm.residuals_ is not None
        assert glm.residuals_.shape == (N,)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_logistic_predict_shape(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        p_hat = glm.predict(X)
        assert p_hat.shape == (N,)

    def test_logistic_predict_in_unit_interval(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        p_hat = glm.predict(X)
        assert np.all(p_hat >= 0) and np.all(p_hat <= 1)

    def test_linear_predict_shape(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        y_hat = glm.predict(X)
        assert y_hat.shape == (N,)

    def test_linear_unpenalized_predict_close_to_ols(self):
        # With no penalty and enough data, predictions should closely track OLS.
        # Use tighter solver settings so the iterative solution is near-exact.
        glm = GLM(family="linear", l1_penalty=0, min_its=500, tol=1e-12).fit(X, Y_LINEAR)
        y_hat = glm.predict(X)
        # OLS predictions
        ols_coef = np.linalg.lstsq(X_INT, Y_LINEAR, rcond=None)[0]
        y_hat_ols = X_INT @ ols_coef
        assert np.allclose(y_hat, y_hat_ols, rtol=1e-3, atol=1e-3)

    def test_predict_before_fit_raises(self):
        glm = GLM(family="logistic", l1_penalty=LAM)
        with pytest.raises(AssertionError):
            glm.predict(X)


# ---------------------------------------------------------------------------
# Active set
# ---------------------------------------------------------------------------

class TestActiveSet:
    def test_active_returns_array(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        active = glm.active()
        assert isinstance(active, np.ndarray)

    def test_active_indices_in_range(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        active = glm.active()
        assert np.all(active >= 0) and np.all(active < P)

    def test_active_excludes_intercept_index(self):
        # active() should never include the intercept (index 0 of beta_)
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        active = glm.active()
        # active indices are relative to X columns, not beta_ (which includes intercept)
        assert np.all(active < P)

    def test_active_no_intercept(self):
        glm = GLM(family="linear", l1_penalty=LAM, intercept=False).fit(X, Y_LINEAR)
        active = glm.active()
        assert np.all(active >= 0) and np.all(active < P)

    def test_no_penalty_selects_more(self):
        glm_penalized = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        glm_unpenalized = GLM(family="linear", l1_penalty=0).fit(X, Y_LINEAR)
        assert len(glm_penalized.active()) <= len(glm_unpenalized.active())

    def test_active_before_fit_raises(self):
        glm = GLM(family="linear", l1_penalty=LAM)
        with pytest.raises(ValueError):
            glm.active()


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------

class TestStandardErrors:
    def test_se_shape_logistic(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        se = glm.se(X)
        assert se.shape == (P + 1,)

    def test_se_shape_linear(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        se = glm.se(X)
        assert se.shape == (P + 1,)

    def test_se_all_positive(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        se = glm.se(X)
        assert np.all(se > 0)

    def test_se_clustered_shape(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        se_cr1 = glm.se(X, clusters=CLUSTERS)
        assert se_cr1.shape == (P + 1,)

    def test_se_clustered_positive(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        se_cr1 = glm.se(X, clusters=CLUSTERS)
        assert np.all(se_cr1 > 0)

    def test_se_before_fit_raises(self):
        glm = GLM(family="linear", l1_penalty=LAM)
        with pytest.raises(ValueError):
            glm.se(X)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

class TestConfInt:
    def test_shape_logistic(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        ci = glm.conf_int(X)
        assert ci.shape == (P + 1, 2)

    def test_shape_linear(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci = glm.conf_int(X)
        assert ci.shape == (P + 1, 2)

    def test_lower_leq_upper(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci = glm.conf_int(X)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_beta_inside_interval(self):
        # Fitted beta_ must lie inside its own confidence interval
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci = glm.conf_int(X)
        assert np.all(glm.beta_ >= ci[:, 0]) and np.all(glm.beta_ <= ci[:, 1])

    def test_wider_at_lower_level(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci_90 = glm.conf_int(X, level=0.90)
        ci_99 = glm.conf_int(X, level=0.99)
        widths_90 = ci_90[:, 1] - ci_90[:, 0]
        widths_99 = ci_99[:, 1] - ci_99[:, 0]
        assert np.all(widths_99 >= widths_90)

    def test_clustered_conf_int_shape(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci = glm.conf_int(X, clusters=CLUSTERS)
        assert ci.shape == (P + 1, 2)

    def test_clustered_lower_leq_upper(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        ci = glm.conf_int(X, clusters=CLUSTERS)
        assert np.all(ci[:, 0] <= ci[:, 1])


# ---------------------------------------------------------------------------
# get_var
# ---------------------------------------------------------------------------

class TestGetVar:
    def test_homogeneous_returns_constant(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        var = glm.get_var(X, Y_LINEAR, error_model="homogeneous")
        assert var.shape == (N,)
        assert np.allclose(var, var[0]), "Homogeneous variance should be constant"
        assert var[0] > 0

    def test_heterogeneous_returns_per_obs(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        var = glm.get_var(X, Y_LINEAR, error_model="heterogeneous")
        assert var.shape == (N,)
        assert np.all(var >= 0)

    def test_none_error_model_same_as_heterogeneous(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        var_het = glm.get_var(X, Y_LINEAR, error_model="heterogeneous")
        var_none = glm.get_var(X, Y_LINEAR, error_model=None)
        assert np.allclose(var_het, var_none)

    def test_clustered_returns_matrix(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        var = glm.get_var(X, Y_LINEAR, error_model="clustered", clusters=CLUSTERS)
        assert var.shape == (CLUSTER_SIZE, CLUSTER_SIZE)

    def test_logistic_heterogeneous_in_range(self):
        glm = GLM(family="logistic", l1_penalty=LAM).fit(X, Y_BIN)
        var = glm.get_var(X, Y_BIN, error_model="heterogeneous")
        assert var.shape == (N,)
        assert np.all(var >= 0) and np.all(var <= 0.25 + 1e-10)

    def test_invalid_error_model_raises(self):
        glm = GLM(family="linear", l1_penalty=LAM).fit(X, Y_LINEAR)
        with pytest.raises(ValueError):
            glm.get_var(X, Y_LINEAR, error_model="bad_model")

    def test_get_var_before_fit_raises(self):
        glm = GLM(family="linear", l1_penalty=LAM)
        with pytest.raises(AssertionError):
            glm.get_var(X, Y_LINEAR)


# ---------------------------------------------------------------------------
# Poisson GLM integration tests
# ---------------------------------------------------------------------------

class TestPoissonGLM:
    def setup_method(self):
        self.glm = GLM(family="poisson", l1_penalty=LAM).fit(X, Y_COUNT)

    def test_predict_positive(self):
        mu = self.glm.predict(X)
        assert mu.shape == (N,)
        assert np.all(mu > 0)

    def test_conf_int_shape(self):
        ci = self.glm.conf_int(X)
        assert ci.shape == (P + 1, 2)

    def test_conf_int_lower_leq_upper(self):
        ci = self.glm.conf_int(X)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_conf_int_beta_inside(self):
        ci = self.glm.conf_int(X)
        assert np.all(self.glm.beta_ >= ci[:, 0])
        assert np.all(self.glm.beta_ <= ci[:, 1])

    def test_conf_int_wider_at_higher_level(self):
        ci_90 = self.glm.conf_int(X, level=0.90)
        ci_99 = self.glm.conf_int(X, level=0.99)
        assert np.all((ci_99[:, 1] - ci_99[:, 0]) >= (ci_90[:, 1] - ci_90[:, 0]))

    def test_se_positive(self):
        se = self.glm.se(X)
        assert se.shape == (P + 1,)
        assert np.all(se > 0)

    def test_get_var_heterogeneous_positive(self):
        var = self.glm.get_var(X, Y_COUNT, error_model="heterogeneous")
        assert var.shape == (N,)
        assert np.all(var > 0)

    def test_active_indices_in_range(self):
        active = self.glm.active()
        assert np.all(active >= 0) and np.all(active < P)

    def test_matches_direct_regreg_construction(self):
        from m_estimation_SI.losses import poisson_loss_smooth
        import regreg.api as rr
        X_int = np.hstack([np.ones((N, 1)), X])
        loss = poisson_loss_smooth(X_int, Y_COUNT)
        penalty = rr.weighted_l1norm([0] + [1] * P, lagrange=LAM)
        problem = rr.simple_problem(loss, penalty)
        beta_direct = problem.solve(min_its=50, tol=1e-8)
        assert np.allclose(self.glm.beta_, beta_direct)


# ---------------------------------------------------------------------------
# Negative Binomial GLM integration tests
# ---------------------------------------------------------------------------

class TestNegBinGLM:
    def setup_method(self):
        self.glm = GLM(
            family="negative_binomial", l1_penalty=LAM, theta=THETA_NB
        ).fit(X, Y_NB)

    def test_predict_positive(self):
        mu = self.glm.predict(X)
        assert mu.shape == (N,)
        assert np.all(mu > 0)

    def test_conf_int_shape(self):
        ci = self.glm.conf_int(X)
        assert ci.shape == (P + 1, 2)

    def test_conf_int_lower_leq_upper(self):
        ci = self.glm.conf_int(X)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_conf_int_beta_inside(self):
        ci = self.glm.conf_int(X)
        assert np.all(self.glm.beta_ >= ci[:, 0])
        assert np.all(self.glm.beta_ <= ci[:, 1])

    def test_conf_int_wider_at_higher_level(self):
        ci_90 = self.glm.conf_int(X, level=0.90)
        ci_99 = self.glm.conf_int(X, level=0.99)
        assert np.all((ci_99[:, 1] - ci_99[:, 0]) >= (ci_90[:, 1] - ci_90[:, 0]))

    def test_se_positive(self):
        se = self.glm.se(X)
        assert se.shape == (P + 1,)
        assert np.all(se > 0)

    def test_get_var_exceeds_poisson(self):
        # NB variance > Poisson variance for any finite theta
        var = self.glm.get_var(X, Y_NB, error_model="heterogeneous")
        mu = self.glm.predict(X)
        assert np.all(var > mu)

    def test_active_indices_in_range(self):
        active = self.glm.active()
        assert np.all(active >= 0) and np.all(active < P)

    def test_matches_direct_regreg_construction(self):
        from m_estimation_SI.losses import negative_binomial_loss_smooth
        import regreg.api as rr
        X_int = np.hstack([np.ones((N, 1)), X])
        loss = negative_binomial_loss_smooth(X_int, Y_NB, THETA_NB)
        penalty = rr.weighted_l1norm([0] + [1] * P, lagrange=LAM)
        problem = rr.simple_problem(loss, penalty)
        beta_direct = problem.solve(min_its=50, tol=1e-8)
        assert np.allclose(self.glm.beta_, beta_direct)

    def test_large_theta_close_to_poisson(self):
        # With huge theta, NB should give nearly the same fit as Poisson
        glm_nb = GLM(family="negative_binomial", l1_penalty=LAM, theta=1e5).fit(X, Y_COUNT)
        glm_p = GLM(family="poisson", l1_penalty=LAM).fit(X, Y_COUNT)
        assert np.allclose(glm_nb.beta_, glm_p.beta_, atol=1e-3)


# ---------------------------------------------------------------------------
# NB IRLS (automatic theta estimation)
# ---------------------------------------------------------------------------

class TestNegBinIRLS:
    """Tests for the two-step IRLS theta estimation path (theta=None)."""

    def test_theta_stored_on_self(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert glm.theta == glm.theta_

    def test_theta_positive(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert glm.theta_ > 0

    def test_theta_in_plausible_range(self):
        # True theta is THETA_NB=3; estimate should be in a wide but sane range
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert 0.1 < glm.theta_ < 100

    def test_theta_stored_as_scalar(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert np.isscalar(glm.theta_) or np.ndim(glm.theta_) == 0

    def test_beta_same_shape_as_fixed_theta(self):
        glm_est = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        glm_fix = GLM(family="negative_binomial", l1_penalty=LAM, theta=THETA_NB).fit(X, Y_NB)
        assert glm_est.beta_.shape == glm_fix.beta_.shape

    def test_conf_int_valid(self):
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        ci = glm.conf_int(X)
        assert ci.shape == (P + 1, 2)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_loss_consistent_with_estimated_theta(self):
        # loss_.theta should match the theta stored on self after fit
        glm = GLM(family="negative_binomial", l1_penalty=LAM).fit(X, Y_NB)
        assert np.isclose(glm.loss_.theta, glm.theta_)

    def test_irls_improves_over_fixed_poisson(self):
        # NB log-likelihood at the IRLS solution should be >= at the Poisson fit
        from m_estimation_SI.losses import negative_binomial_loss_smooth
        glm_nb = GLM(family="negative_binomial", l1_penalty=0).fit(X, Y_NB)
        glm_p = GLM(family="poisson", l1_penalty=0).fit(X, Y_NB)
        X_fit = np.hstack([np.ones((N, 1)), X])
        loss_nb = negative_binomial_loss_smooth(X_fit, Y_NB, glm_nb.theta_)
        # NB nll at NB solution <= NB nll at Poisson solution (NB MLE minimises NB loss)
        f_nb = loss_nb.smooth_objective(glm_nb.beta_, mode="func")
        f_p = loss_nb.smooth_objective(glm_p.beta_, mode="func")
        assert f_nb <= f_p + 1e-6
