from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._base import AbstractRealIsotonicRegression
from .curves import PiecewiseLinearIsotonicCurve


__all__ = ['LpIsotonicRegression']


class LpIsotonicRegression(AbstractRealIsotonicRegression):
    def __init__(self, npoints, penalty={2: 1}, increasing=True, cut_algo='quantile', curve_algo=PiecewiseLinearIsotonicCurve):
        super().__init__(npoints, increasing=increasing, cut_algo=cut_algo, curve_algo=curve_algo)
        for p in penalty:
            assert (p >= 1) and (not np.isinf(p)), 'Can only have an l^p penalty for p \in [1,\infty)'
            assert (penalty[p] > 0), 'coefficients on penalties must be positive'
        self.penalty = penalty

    def _check_x_y(self, X, y):
        assert np.all(np.isfinite(X)), "All x-values must be finite"
        assert np.all(np.isfinite(y)), "All y-values must be finite"

    def _err_func(self, x_cuts, X, y):
        def err(alpha):
            gamma = self.gamma_of_alpha(alpha)
            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            y_p = curve.f(X)
            result = 0
            for pwr in self.penalty:
                result += (self.penalty[pwr]*np.power(np.abs(y_p-y), pwr)).sum()
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
            for pwr in self.penalty:
                if pwr == 1:
                    dE_dgamma += np.sign(delta)
                else:
                    dE_dgamma += self.penalty[pwr] * pwr * np.power(np.abs(delta), pwr-1) * np.sign(delta)
            dE_dgamma = curve.grad_y(X) @ dE_dgamma
            result = self.grad_gamma_of_alpha(alpha) @ dE_dgamma / N
            return result
        return grad_err
