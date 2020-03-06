import numpy as np
from scipy import interpolate
from scipy.sparse import csc_matrix
import abc

__all__ = ['PiecewiseConstantIsotonicCurve', 'PiecewiseLinearIsotonicCurve']


class AbstractIsotonicCurve(metaclass=abc.ABCMeta):
    def __init__(self, x, y, increasing=True):
        x = np.array(x)
        y = np.array(y)
        self.__check_monotonic(x, 'x', True)
        assert (np.all(x[1:] > x[0:-1])), "Each x-value must be larger than the previous one."
        self.x = x
        self.__check_monotonic(y, 'y', increasing)
        self.y = y
        self.increasing = increasing
        self._build_f()

    def __check_monotonic(self, x, varname, increasing):
        if increasing:
            if (not np.all(x[:-1] <= x[1:])):
                raise ValueError("Input " + varname + " was not monotonically increasing, but should have been.")
        else:
            if (np.all(x[:-1] <= x[1:])):
                raise ValueError("Input " + varname + " was not monotonically decreasing, but should have been.")

    @abc.abstractmethod
    def _build_f(self):
        pass

    @abc.abstractmethod
    def grad_y(self, x):
        pass

    def f(self, x):
        return self.f_(x)


class PiecewiseConstantIsotonicCurve(AbstractIsotonicCurve):
    """
    For the set of points [(x[i], y[i])], this curve will
    return y[i] corresponding to the largest x[i] < x.

    I.e. piecewise constant.
    """
    def _build_f(self):
        self.f_ = interpolate.interp1d(self.x, self.y, kind='previous', fill_value='extrapolate')

    def grad_y(self, x):
        bins = np.digitize(x, self.x) - 1
        result = csc_matrix(
            (np.ones(shape=(len(x),)), (bins, np.arange(0,len(x)))),
            shape=(len(self.y), len(x)))
        return result


class PiecewiseLinearIsotonicCurve(AbstractIsotonicCurve):
    """
    Returns an isotonic curve that is continuous.
    """
    def _build_f(self):
        self.f_ = interpolate.interp1d(self.x, self.y, kind='linear', fill_value=(self.y[0], self.y[-1]), bounds_error=False)

    def grad_y(self, x):
        bins = np.digitize(x, self.x) - 1
        bins = np.maximum(bins, 0)
        bins_p1 = bins+1
        bins_p1 = np.minimum(bins_p1, len(self.x)-1)

        # The value of the non-exterior point x is `alpha y[b] + (1-alpha)*y[b+1]`.
        # Here, alpha = (x-x[b])/(x[b+1]-x[b])
        # Thus, the grad is [0, ..., alpha, (1-alpha), ...]

        alpha = np.nan_to_num((self.x[bins_p1] - x) / (self.x[bins_p1] - self.x[bins]), 1)
        alpha[x > self.x[-1]] = 1.0
        alpha[x <= self.x[0]] = 1.0

        result = csc_matrix(
            (alpha, (bins, np.arange(0,len(x)))),
            shape=(len(self.y), len(x)))
        result2 = csc_matrix(
            (1-alpha, (bins_p1, np.arange(0,len(x)))),
             shape=(len(self.y), len(x))
        )
        return result + result2


class PChipIsotonicCurve(AbstractIsotonicCurve):
    """
    Builds a smooth isotonic curve based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

    This curve will be continuous, monotonic, and differentiable at all internal points.

    This is experimental and cannot be used, since we do not have it's gradient yet.
    """
    def _build_f(self):
        # By default the PchipInterpolator does not have a zero derivative
        # at the right endpoint.
        # To force it to do this, we extend the array by one and make
        # the right-most segment flat.
        xx = np.zeros(shape=(self.x.shape[0]+1,))
        xx[0:-1] = self.x
        xx[-1] = xx[-2]+1
        yy = np.zeros(shape=(self.y.shape[0]+1,))
        yy[0:-1] = self.y
        yy[-1] = yy[-2]
        self.interpolator_ = interpolate.PchipInterpolator(xx, yy, extrapolate=False)

    def f_(self, x):
        result = self.interpolator_(x)
        result[x <= self.x.min()] = self.y[0]
        result[x >= self.x.max()] = self.y[-1]
        return result
