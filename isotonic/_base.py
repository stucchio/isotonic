from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.optimize import minimize
import numpy as np
from .curves import PiecewiseLinearIsotonicCurve
import abc
import logging
_log = logging.getLogger(__name__)


__all__ = ['AbstractProbabilityIsotonicRegression', 'AbstractRealIsotonicRegression']


class _BaseIsotonicFit(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gamma_of_alpha(self, alpha):
        pass

    @abc.abstractmethod
    def grad_gamma_of_alpha(self, alpha):
        pass

    @abc.abstractmethod
    def parameterization_dim(self, N):
        """
        How many dimensions the parameterization is as a function of the number
        of nodes.
        """
        pass

    def fit(self, X, y):
        self._check_x_y(X,y)
        if not self.increasing:  # We handle decreasing data simply by multiplying by -1 internally
            X = X * (-1)

        if self.cut_algo == 'quantile':
            x_cuts = np.quantile(X, np.arange(0 + 0.5/self.npoints, 1 + 0.5/self.npoints, 1/self.npoints))
        else:
            x_cuts = X.min() + (X.max() - X.min())*np.arange(0,1,1/self.npoints)
        alpha = np.zeros(shape=(int(self.parameterization_dim(self.npoints)),))

        err = self._err_func(x_cuts, X, y)
        grad_err = self._grad_err_func(x_cuts, X, y)

        for (i, method) in enumerate(['CG', 'BFGS', 'Nelder-Mead']): # We iterate over possible algorithms, since occasionally we don't get convergence on the first try.
            min_result = minimize(err, x0=alpha, method=method, jac=grad_err)
            if min_result.success:
                _log.info("Succeeded at isotonic fit using method %s at attempt %s", method, i)
                break
            elif (not min_result.success):
                _log.warning("Did not achieve success in minimization with method %s at iteration %s.", method, i)
                _log.warning("Minimization result: %s", min_result)
                alpha = min_result.x  # We assume that the previous attempt had some improvement, at least
        if not min_result.success:
            _log.warning("Did not achieve success with any optimization method.")
        self.curve_ = self.curve_algo(x_cuts, self.gamma_of_alpha(min_result.x), increasing=True)
        self.fitted_ = True
        return self

    def transform(self, X):
        assert self.fitted_
        if self.increasing:
            return self.curve_.f(X)
        else:
            return self.curve_.f(-1*X)

    def predict_proba(self, X):
        return self.transform(X)


class AbstractProbabilityIsotonicRegression(ClassifierMixin, TransformerMixin, BaseEstimator, _BaseIsotonicFit, metaclass=abc.ABCMeta):
    """
    Abstract base class for isotonic regression where the resulting function is interpreted as a probability.

    I.e., this enforces the constraint that 0 < curve.f(x) < 1.
    """
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

    def parameterization_dim(self, N):
        """
        How many dimensions the parameterization is as a function of the number
        of nodes.
        """
        return N+1

    def gamma_of_alpha(self, alpha):
        """
        This parameterization generates a sequence of variables, 0 < ... gamma[i] < gamma[i+1] < 1
        for any real vector alpha.

        These variables are constrained to lie in (0,1) (exclusive of endpoints) by the parameterization.

        The dimension of gamma is one less than that of alpha.
        """
        gamma = np.exp(alpha[:-1]).cumsum()
        gamma /= (np.exp(alpha).sum())
        return gamma

    def grad_gamma_of_alpha(self, alpha):
        """
        The gradient of `self.gamma_of_alpha`.
        """
        exp_alpha = np.exp(alpha)

        J1 = np.triu(np.ones(shape=(len(alpha), len(alpha)-1)))
        J1 *= exp_alpha.sum()

        J2 = np.zeros(shape=(len(alpha), len(alpha)-1))
        J2 -= exp_alpha.cumsum()[np.newaxis,0:-1]
        return (J1 + J2)* exp_alpha[:,np.newaxis] / (np.power(exp_alpha.sum(), 2))

    @abc.abstractmethod
    def _check_x_y(self, X, y):
        return None


class AbstractRealIsotonicRegression(ClassifierMixin, TransformerMixin, BaseEstimator, _BaseIsotonicFit, metaclass=abc.ABCMeta):
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

    def parameterization_dim(self, N):
        """
        How many dimensions the parameterization is as a function of the number
        of nodes.
        """
        return N

    @abc.abstractmethod
    def _check_x_y(self, X, y):
        return None

    def gamma_of_alpha(self, alpha):
        """
        This parameterization generates a sequence of variables, gamma[0] < ... gamma[i] < gamma[i+1]
        for any real vector alpha.

        These variables are unconstrained apart from that.

        The dimension of gamma is one less than that of alpha.
        """
        gamma = np.full(shape=(len(alpha),), fill_value=alpha[0])
        gamma[1:] += np.exp(alpha[1:]).cumsum()
        return gamma

    def grad_gamma_of_alpha(self, alpha):
        """
        The gradient of `self.gamma_of_alpha`.
        """
        J = np.triu(np.ones(shape=(len(alpha), len(alpha))))
        J[0,:] = 1
        J[1:,:] *= np.exp(alpha[1:, np.newaxis])
        return J
