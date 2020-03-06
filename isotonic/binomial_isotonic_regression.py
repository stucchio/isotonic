from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._parameterization import gamma_of_alpha, grad_gamma_of_alpha
from ._base import AbstractIsotonicRegression
from .curves import PiecewiseLinearIsotonicCurve


__all__ = ['BinomialIsotonicRegression']


class BinomialIsotonicRegression(AbstractIsotonicRegression):
    def _err_func(self, x_cuts, X, y):
        def err(alpha):
            gamma = gamma_of_alpha(alpha)
            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            p = curve.f(X)
            result = -1*(np.log(p[y == 1]).sum() + np.log(1-p[y==0]).sum())
            return result / len(X)
        return err

    def _grad_err_func(self, x_cuts, X, y):
        def grad_err(alpha):
            gamma = gamma_of_alpha(alpha)

            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            p = curve.f(X)
            dE_dgamma = np.zeros(shape=(len(X),))
            dE_dgamma[y == 1] = 1.0/p[y == 1]
            dE_dgamma[y == 0] = -1.0/(1-p[y == 0])
            dE_dgamma = curve.grad_y(X) @ dE_dgamma
            return -1*grad_gamma_of_alpha(alpha) @ dE_dgamma / len(X)
        return grad_err
