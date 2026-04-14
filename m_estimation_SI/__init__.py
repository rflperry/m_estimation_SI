"""m_estimation_SI — selective inference for penalized M-estimators.

The package provides:

* :class:`GLM` — penalized logistic and linear regression with HC1/CR1
  sandwich standard errors and Wald confidence intervals.
* :func:`logistic_group_instance` — synthetic data generator for
  group-lasso selective inference experiments.
* :class:`logistic_loss_smooth`, :class:`least_squares_loss_smooth` —
  regreg-compatible mean-scaled loss functions.
"""

from .losses import logistic_loss_smooth, least_squares_loss_smooth  # noqa: F401
from .penalized_glm import GLM  # noqa: F401
from .simulation import logistic_group_instance  # noqa: F401

__all__ = [
    "GLM",
    "logistic_group_instance",
    "logistic_loss_smooth",
    "least_squares_loss_smooth",
]
