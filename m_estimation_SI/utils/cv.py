import numpy as np
from m_estimation_SI import GLM
from sklearn.linear_model import LassoCV

def select_lambda_by_mse_for_lasso(
    family,
    X,
    Y,
    intercept,
    base_penalty,
    num_candidates=10,
    scale=4.0,
):
    """Select a lasso penalty by minimizing in-sample mean squared error."""
    if base_penalty <= 0:
        return 0.0

    n, _ = X.shape
    penalty_candidates = np.geomspace(base_penalty / scale, base_penalty * scale, num_candidates)
    best_mse = np.inf
    best_penalty = base_penalty

    for penalty in penalty_candidates:
        try:
            model = GLM(family=family, l1_penalty=penalty, intercept=intercept).fit(X, Y)
        except Exception:
            continue

        if hasattr(model, "predict"):
            pred = model.predict(X)
        else:
            if intercept:
                pred = np.hstack([np.ones((n, 1)), X]) @ model.beta_
            else:
                pred = X @ model.beta_

        mse = np.mean((Y - pred) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_penalty = penalty

    return best_penalty


def select_lambda_lassocv(X, Y, intercept=True, cv=10, base_penalty=None, num_candidates=100, **kwargs):
    """Select a lasso penalty via sklearn's LassoCV.

    sklearn's Lasso objective uses (1/(2n))||y - Xw||^2 + alpha*||w||_1,
    while GLM's regreg-based loss omits the 1/2 factor, so the selected
    alpha is doubled to match GLM's l1_penalty scaling.
    """
    alphas = np.geomspace(base_penalty * 0.01, base_penalty * 5, num_candidates)
    lasso_cv = LassoCV(fit_intercept=intercept, cv=cv, alphas = alphas, **kwargs).fit(X, Y)
    return 2 * lasso_cv.alpha_
