from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._base import AbstractProbabilityIsotonicRegression
from .curves import PiecewiseLinearIsotonicCurve


__all__ = ['BinomialIsotonicRegression']


class BinomialIsotonicRegression(AbstractProbabilityIsotonicRegression):
    def _check_x_y(self, X, y):
        assert (((y == 0) | (y == 1)).all()), "All y-values must be either 0 or 1"

    def _err_func(self, x_cuts, X, y):
        def err(alpha):
            gamma = self.gamma_of_alpha(alpha)
            curve = self.curve_algo(x=x_cuts, y=gamma)
            p = curve.f(X)
            result = -1*(np.log(p[y == 1]).sum() + np.log(1-p[y==0]).sum())
            return result / len(X)
        return err

    def _grad_err_func(self, x_cuts, X, y):
        grad_y = []  # Part of terrible performance hack

        def grad_err(alpha):
            gamma = self.gamma_of_alpha(alpha)

            curve = self.curve_algo(x=x_cuts, y=gamma)
            p = curve.f(X)
            dE_dgamma = np.zeros(shape=(len(X),))
            dE_dgamma[y == 1] = 1.0/p[y == 1]
            dE_dgamma[y == 0] = -1.0/(1-p[y == 0])
            if len(grad_y) == 0: # Terrible performance hack
                grad_y.append(curve.grad_y(X))  # This value depends only on x_cuts, so if we calculate it once we don't need to recalculate it
            dE_dgamma = grad_y[0] @ dE_dgamma
            return -1*self.grad_gamma_of_alpha(alpha) @ dE_dgamma / len(X)
        return grad_err
