"""m_estimation_SI — selective inference for penalized M-estimators.

The package provides:

* :class:`GLM` — penalized logistic, linear, Poisson, and negative binomial
  regression with HC1/CR1 sandwich standard errors and Wald confidence
  intervals.
* :func:`logistic_group_instance` — synthetic data generator for
  group-lasso selective inference experiments.
* :class:`logistic_loss_smooth`, :class:`least_squares_loss_smooth`,
  :class:`poisson_loss_smooth`, :class:`negative_binomial_loss_smooth` —
  regreg-compatible mean-scaled loss functions.
"""

from .losses import (  # noqa: F401
    logistic_loss_smooth,
    least_squares_loss_smooth,
    poisson_loss_smooth,
    negative_binomial_loss_smooth,
)
from .penalized_glm import GLM  # noqa: F401
from .simulation import logistic_group_instance  # noqa: F401

__all__ = [
    "GLM",
    "logistic_group_instance",
    "logistic_loss_smooth",
    "least_squares_loss_smooth",
    "poisson_loss_smooth",
    "negative_binomial_loss_smooth",
]
