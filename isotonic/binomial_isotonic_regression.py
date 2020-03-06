from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from ._parameterization import gamma_of_alpha, grad_gamma_of_alpha
from ._base import AbstractIsotonicRegression
from .curves import PiecewiseLinearIsotonicCurve


__all__ = ['BinomialIsotonicRegression']


class BinomialIsotonicRegression(AbstractIsotonicRegression):
    def __init__(self, npoints, increasing=True, cut_algo='quantile', curve_algo=PiecewiseLinearIsotonicCurve):
        assert (cut_algo in ['quantile', 'uniform'])
        self.npoints = npoints
        self.cut_algo = cut_algo
        self.increasing = increasing
        self.curve_algo = curve_algo
        self.fitted_ = False

    def fit(self, X, y):
        if self.cut_algo == 'quantile':
            x_cuts = np.quantile(X, np.arange(0,1,1/self.npoints))
        else:
            x_cuts = X.min() + (X.max() - X.min())*np.arange(0,1,1/self.npoints)
        alpha = np.zeros(shape=(int(self.npoints+1),))

        def err(alpha):
            gamma = gamma_of_alpha(alpha)
            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            p = curve.f(X)
            result = -1*(np.log(p[y == 1]).sum() + np.log(1-p[y==0]).sum())
            return result / len(X)

        def grad_err(alpha):
            gamma = gamma_of_alpha(alpha)

            curve = self.curve_algo(x=x_cuts, y=gamma, increasing=self.increasing)
            p = curve.f(X)
            dE_dgamma = np.zeros(shape=(len(X),))
            dE_dgamma[y == 1] = 1.0/p[y == 1]
            dE_dgamma[y == 0] = -1.0/(1-p[y == 0])
            dE_dgamma = curve.grad_y(X) @ dE_dgamma
            return -1*grad_gamma_of_alpha(alpha) @ dE_dgamma / len(X)

        min_result = minimize(err, x0=alpha, method='CG', jac=grad_err)
        print(min_result)
        self.curve_ = self.curve_algo(x_cuts, gamma_of_alpha(min_result.x), increasing=self.increasing)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        assert self.fitted_
        return self.curve_.f(X)