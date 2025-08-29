import numpy as np
from regreg.smooth import smooth_atom
from scipy.special import expit

class logistic_loss_smooth(smooth_atom):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        n, p = X.shape

        if (sum(y) > n or sum(y) < 0):
            raise ValueError("Sum of y is outside [0, n], problem is not convex.")

        # Call parent constructor with dimension p
        super().__init__(shape=(p,))

    def smooth_objective(self, beta, mode='both', check_feasibility=None):
        Xbeta = self.X @ beta
        # logistic log likelihood contribution
        logexp = np.logaddexp(0, Xbeta)   # log(1+exp(xβ))
        f = (logexp - self.y * Xbeta).sum()

        if mode == 'func':
            return f
        elif mode == 'grad':
            # gradient = X^T (pi - y)
            pi = expit(Xbeta)
            g = self.X.T @ (pi - self.y)
            return g
        elif mode == 'both':
            pi = expit(Xbeta)
            g = self.X.T @ (pi - self.y)
            return f, g
        else:
            raise ValueError("mode must be 'func', 'grad', or 'both'")
