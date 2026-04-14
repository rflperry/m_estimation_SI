import pytest
import numpy as np
from m_estimation_SI import GLM
import regreg.api as rr


solve_kwargs = {'min_its':100, 'tol':1.e-10}
n, p = 100, 10
lam = np.sqrt(2*np.log(p)/n)

np.random.seed(1)
X = np.random.randn(n, p)
X /= np.linalg.norm(X, axis=0)  # normalize columns
X_intercept = np.hstack([np.ones((n, 1)), X])  # add intercept

beta_true = np.zeros(p+1); beta_true[:3] = [3, 2, 1]
probs = 1/(1+np.exp(-(X_intercept @ beta_true)))
y = np.random.binomial(1, probs)


def test_logistic_matches_builtin():
    # GLM class should produce the same result as directly building a regreg
    # problem with logistic_loss_smooth (our mean-scaled custom loss).
    from m_estimation_SI.losses import logistic_loss_smooth
    loss = logistic_loss_smooth(X_intercept, y)
    penalty = rr.weighted_l1norm([0] + [1]*p, lagrange=lam)
    problem = rr.simple_problem(loss, penalty)
    beta_hat_direct = problem.solve(**solve_kwargs)

    beta_hat = GLM(family="logistic", l1_penalty=lam, **solve_kwargs).fit(X, y).beta_

    assert np.allclose(beta_hat, beta_hat_direct)

def test_logistic_affine():
    beta_hat = GLM(family="logistic", l1_penalty=lam, affine_penalty = np.ones(p+1)*0.1, **solve_kwargs).fit(X, y).beta_
    assert len(beta_hat) == p+1

def test_logistic_confs():
    glm = GLM(family="logistic", l1_penalty=lam, **solve_kwargs).fit(X, y)
    conf_int = glm.conf_int(X)
    assert conf_int.shape == (p+1, 2)
    assert np.all(conf_int[:,1] >= conf_int[:,0])
    
def test_logistic_confs_level():
    # Wider level should produce wider intervals
    glm = GLM(family="logistic", l1_penalty=lam, **solve_kwargs).fit(X, y)
    ci_90 = glm.conf_int(X, level=0.90)
    ci_99 = glm.conf_int(X, level=0.99)
    assert np.all((ci_99[:, 1] - ci_99[:, 0]) >= (ci_90[:, 1] - ci_90[:, 0]))
