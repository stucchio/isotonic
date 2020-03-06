from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._base import AbstractRealIsotonicRegression
from .curves import PiecewiseLinearIsotonicCurve


__all__ = ['LpIsotonicRegression']


class LpIsotonicRegression(AbstractRealIsotonicRegression):
    def __init__(self, npoints, power=2, increasing=True, cut_algo='quantile', curve_algo=PiecewiseLinearIsotonicCurve):
        super().__init__(npoints, increasing=increasing, cut_algo=cut_algo, curve_algo=curve_algo)
        assert (power >= 1), "Power must be bigger than or equal to 1"
        self.power = power

    def _check_x_y(self, X, y):
        assert np.all(np.isfinite(X)), "All x-values must be finite"
        assert np.all(np.isfinite(y)), "All y-values must be finite"

    def _err_func(self, x_cuts, X, y):
        def err(alpha):
            gamma = self.gamma_of_alpha(alpha)
            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            y_p = curve.f(X)
            result = 0
            result += np.power(np.abs(y_p-y), self.power).sum()
            return result / len(X)
        return err

    def _grad_err_func(self, x_cuts, X, y):
        N = len(X)
        def grad_err(alpha):
            gamma = self.gamma_of_alpha(alpha)

            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            y_p = curve.f(X)
            delta = y_p - y
            dE_dgamma = np.zeros(shape=(N,))
            if self.power == 1:
                dE_dgamma += np.sign(delta)
            else:
                dE_dgamma += self.power * np.power(np.abs(delta), self.power-1) * np.sign(delta)
            dE_dgamma = curve.grad_y(X) @ dE_dgamma
            result = self.grad_gamma_of_alpha(alpha) @ dE_dgamma / N
            return result
        return grad_err
