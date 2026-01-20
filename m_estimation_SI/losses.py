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
        f = np.mean(logexp - self.y * Xbeta)

        if mode == 'func':
            return f
        elif mode == 'grad':
            # gradient = X^T (pi - y)
            pi = expit(Xbeta)
            g = self.X.T @ (pi - self.y) / self.X.shape[0]
            return g
        elif mode == 'both':
            pi = expit(Xbeta)
            g = self.X.T @ (pi - self.y) / self.X.shape[0]
            return f, g
        else:
            raise ValueError("mode must be 'func', 'grad', or 'both'")

    def predict_(self, X, beta):
        return expit(X @ beta)
   
    def get_hessian(self, X, beta):
        mu = self.predict_(X, beta)
        W = np.diag(mu * (1 - mu))
        return X.T @ W @ X / X.shape[0]

    def get_var_(self, X, beta, y=None):
        mu = self.predict_(X, beta)
        return mu * (1 - mu)


class least_squares_loss_smooth(smooth_atom):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        _, p = X.shape

        # Call parent constructor with dimension p
        super().__init__(shape=(p,))

    def smooth_objective(self, beta, mode='both', check_feasibility=None):
        n = self.X.shape[0]
        residuals = self.y - self.X @ beta
        f = 0.5 * np.mean(residuals ** 2)

        if mode == 'func':
            return f
        elif mode == 'grad':
            g = -self.X.T @ residuals / n
            return g
        elif mode == 'both':
            g = -self.X.T @ residuals / n
            return f, g
        else:
            raise ValueError("mode must be 'func', 'grad', or 'both'")
        
    def predict_(self, X, beta):
        return X @ beta
    
    def get_hessian(self, X, beta=None):
        return X.T @ X / X.shape[0]
    
    def get_var_(self, X, beta, y=None):
        residuals = y - X @ beta
        return residuals ** 2
