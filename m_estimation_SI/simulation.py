"""Synthetic data generators for group-lasso selective inference experiments."""

import numpy as np

_cov_cache = {}


def _design(n: int, p: int, rho: float, equicorrelated: bool):
    """Draw a design matrix with equicorrelated or AR(1) population covariance.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of continuous features.
    rho : float
        Equicorrelation or AR(1) parameter in ``[0, 1)``.
    equicorrelated : bool
        If ``True``, use equicorrelated covariance; otherwise use AR(1).

    Returns
    -------
    X : ndarray of shape (n, p)
    sigmaX : ndarray of shape (p, p)
        Population covariance of the rows.
    cholX : ndarray of shape (p, p)
        Cholesky factor of *sigmaX*.
    """
    if equicorrelated:
        def _equi(rho, p):
            if ("equi", p, rho) not in _cov_cache:
                sigmaX = (1 - rho) * np.identity(p) + rho * np.ones((p, p))
                _cov_cache[("equi", p, rho)] = sigmaX, np.linalg.cholesky(sigmaX)
            return _cov_cache[("equi", p, rho)]

        sigmaX, cholX = _equi(rho, p)
        X = np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(
            rho
        ) * np.random.standard_normal(n)[:, None]
    else:
        def _ar1(rho, p):
            if ("AR1", p, rho) not in _cov_cache:
                idx = np.arange(p)
                cov = rho ** np.abs(np.subtract.outer(idx, idx))
                _cov_cache[("AR1", p, rho)] = cov, np.linalg.cholesky(cov)
            return _cov_cache[("AR1", p, rho)]

        sigmaX, cholX = _ar1(rho, p)
        X = np.random.standard_normal((n, p)) @ cholX.T

    return X, sigmaX, cholX


def logistic_group_instance(
    n: int = 100,
    p: int = 200,
    sgroup: int = 7,
    ndiscrete: int = 4,
    nlevels: int = None,
    sdiscrete: int = 2,
    rho: float = 0.3,
    signal=7,
    random_signs: bool = False,
    scale: bool = True,
    center: bool = True,
    groups=np.arange(20).repeat(10),
    equicorrelated: bool = True,
):
    """Generate a synthetic group-lasso instance for logistic regression.

    Produces a binary response *Y* and a design matrix *X* whose columns
    are partitioned into groups, with a sparse true signal concentrated
    in ``sgroup`` groups.  Columns are scaled so that
    ``norm(X[:, j]) = 1 / sqrt(n)`` for continuous features.

    Parameters
    ----------
    n : int, default 100
        Number of observations.
    p : int, default 200
        Total number of features (must equal ``len(groups)``).
    sgroup : int or list of int, default 7
        Number of active (signal-carrying) groups.  If a list, specifies
        which group labels are active.
    ndiscrete : int, default 4
        Number of active groups that encode discrete (categorical) variables.
        Must satisfy ``ndiscrete <= sgroup``.
    nlevels : int or None
        Number of levels for each discrete variable.  Each discrete variable
        contributes ``nlevels - 1`` indicator columns (dummy coding).
    sdiscrete : int, default 2
        Number of active discrete groups (subset of *sgroup*).
    rho : float, default 0.3
        Equicorrelation or AR(1) parameter for continuous features.
    signal : float or tuple of float, default 7
        Signal magnitude for active coefficients before the ``1/sqrt(n)``
        normalisation.  If a 2-tuple ``(lo, hi)``, active coefficients are
        spaced linearly between *lo* and *hi*.
    random_signs : bool, default False
        If ``True``, randomly flip the signs of active coefficients.
    scale : bool, default True
        Standardise columns of *X*.
    center : bool, default True
        Centre columns of *X* (continuous features only).
    groups : array-like of shape (p,), default ``np.arange(20).repeat(10)``
        Group assignment for each feature column.
    equicorrelated : bool, default True
        If ``True``, continuous features have equicorrelated population
        covariance; otherwise AR(1).

    Returns
    -------
    X : ndarray of shape (n, p)
        Design matrix.
    Y : ndarray of shape (n,)
        Binary response vector in ``{0, 1}``.
    beta : ndarray of shape (p,)
        True coefficient vector (zero for inactive features).
    active : ndarray of int
        Indices of the non-zero entries of *beta*.
    sigmaX : ndarray of shape (p_cts, p_cts) or None
        Population covariance of the continuous columns.  ``None`` when
        the design is entirely discrete.

    Notes
    -----
    The continuous features are scaled by ``1 / sqrt(n)`` so that
    ``norm(X[:, j]) ≈ 1`` (i.e. columns have unit sample norm), and *beta*
    is scaled reciprocally so that the linear predictor ``X @ beta`` has
    the same magnitude regardless of *n*.

    References
    ----------
    Adapted from the PoSI-GroupLASSO simulation code by Yiling Huang et al.
    https://github.com/yiling-h/PoSI-GroupLASSO
    """
    all_cts = ndiscrete == 0
    all_discrete = (not all_cts) and (ndiscrete * (nlevels - 1) == p)

    # --- discrete variable generation ---

    def _gen_one_discrete():
        probs = np.ones(nlevels) / nlevels
        sample = np.random.choice(np.arange(nlevels), n, p=probs)
        Z = np.zeros((n, nlevels - 1))
        for level in range(1, nlevels):
            Z[:, level - 1] = (sample == level).astype(float)
        return Z

    def _gen_discrete():
        return np.concatenate([_gen_one_discrete() for _ in range(ndiscrete)], axis=1)

    if not all_cts:
        X_indi = _gen_discrete()
        if scale:
            X_indi /= (X_indi.std(0) * np.sqrt(n))[None, :]

    # --- true coefficient vector ---

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)
    group_labels = np.unique(groups)

    if isinstance(sgroup, list):
        group_active = sgroup
    elif all_cts:
        group_active = np.random.choice(group_labels, sgroup, replace=False)
    elif all_discrete:
        group_active = np.random.choice(group_labels, sdiscrete, replace=False)
    else:
        disc_active = np.random.choice(np.arange(ndiscrete), sdiscrete, replace=False)
        cts_groups = np.setdiff1d(group_labels, np.arange(ndiscrete))
        cts_active = np.random.choice(cts_groups, sgroup - sdiscrete, replace=False)
        group_active = np.append(disc_active, cts_active)

    active_mask = np.isin(groups, group_active)
    beta[active_mask] = (
        signal[0]
        if signal.shape == (1,)
        else np.linspace(signal[0], signal[1], active_mask.sum())
    )
    if random_signs:
        beta[active_mask] *= 2 * np.random.binomial(1, 0.5, active_mask.sum()) - 1
    beta /= np.sqrt(n)

    # --- design matrix ---

    if all_discrete:
        X = X_indi
        sigmaX = None
    elif all_cts:
        X, sigmaX = _design(n, p, rho, equicorrelated)[:2]
        if center:
            X -= X.mean(0)
        if scale:
            X /= np.sqrt(n)
            beta *= np.sqrt(n)
            sigmaX = sigmaX / n
    else:
        p_cts = p - ndiscrete * (nlevels - 1)
        X_cts, sigmaX = _design(n, p_cts, rho, equicorrelated)[:2]
        if center:
            X_cts -= X_cts.mean(0)
        if scale:
            X_cts /= np.sqrt(n)
            beta *= np.sqrt(n)
            sigmaX = sigmaX / n
        X = np.concatenate([X_indi, X_cts], axis=1)

    # --- response ---

    eta = X @ beta
    Y = np.random.binomial(1, np.exp(eta) / (1 + np.exp(eta)))

    return X, Y, beta, np.nonzero(active_mask)[0], sigmaX
