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
            p = curve.f(X)
            result = 0
            for pwr in self.penalty:
                result += (self.penalty[pwr]*np.power(np.abs(p-y), pwr)).sum()
            return result / len(X)
        return err

    def _grad_err_func(self, x_cuts, X, y):
        def grad_err(alpha):
            gamma = self.gamma_of_alpha(alpha)

            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            p = curve.f(X)
            dE_dgamma = np.zeros(shape=(len(X),))
            for pwr in self.penalty:
                dE_dgamma += pwr*np.power(np.abs(p-y), pwr)
            dE_dgamma = curve.grad_y(X) @ dE_dgamma
            return -1*self.grad_gamma_of_alpha(alpha) @ dE_dgamma / len(X)
        return grad_err
