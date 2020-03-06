from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._parameterization import gamma_of_alpha, grad_gamma_of_alpha
from .curves import PiecewiseLinearIsotonicCurve
import abc


__all__ = ['AbstractIsotonicRegression']


class AbstractIsotonicRegression(ClassifierMixin, TransformerMixin, BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, npoints, increasing=True, cut_algo='quantile', curve_algo=PiecewiseLinearIsotonicCurve):
        assert (cut_algo in ['quantile', 'uniform'])
        self.npoints = npoints
        self.cut_algo = cut_algo
        self.increasing = increasing
        self.curve_algo = curve_algo
        self.fitted_ = False

    @abc.abstractmethod
    def _err_func(self, x_cuts, X,y):
        pass

    @abc.abstractmethod
    def _grad_err_func(self, x_cuts, X,y):
        pass

    @abc.abstractmethod
    def _check_x_y(self, X, y):
        return None

    def fit(self, X, y):
        self._check_x_y(X,y)

        if self.cut_algo == 'quantile':
            x_cuts = np.quantile(X, np.arange(0,1,1/self.npoints))
        else:
            x_cuts = X.min() + (X.max() - X.min())*np.arange(0,1,1/self.npoints)
        alpha = np.zeros(shape=(int(self.npoints+1),))

        err = self._err_func(x_cuts, X, y)
        grad_err = self._grad_err_func(x_cuts, X, y)

        min_result = minimize(err, x0=alpha, method='CG', jac=grad_err)
        self.curve_ = self.curve_algo(x_cuts, gamma_of_alpha(min_result.x), increasing=self.increasing)
        self.fitted_ = True
        return self

    def transform(self, X):
        assert self.fitted_
        return self.curve_.f(X)

    def predict_proba(self, X):
        return self.transform(X)
