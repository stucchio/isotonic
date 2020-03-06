import numpy as np

__all__ = ['gamma_of_alpha', 'grad_gamma_of_alpha']


def gamma_of_alpha(alpha):
    """
    This parameterization generates a sequence of variables, 0 < ... gamma[i] < gamma[i+1] < 1
    for any real vector alpha.

    The dimension of gamma is one less than that of alpha.
    """
    gamma = np.exp(alpha[:-1]).cumsum()
    gamma /= (np.exp(alpha).sum())
    return gamma


def grad_gamma_of_alpha(alpha):
    """
    The gradient of `gamma_of_alpha`.
    """
    exp_alpha = np.exp(alpha)

    J1 = np.triu(np.ones(shape=(len(alpha), len(alpha)-1)))
    J1 *= exp_alpha.sum()

    J2 = np.zeros(shape=(len(alpha), len(alpha)-1))
    J2 -= exp_alpha.cumsum()[np.newaxis,0:-1]
    return (J1 + J2)* exp_alpha[:,np.newaxis] / (np.power(exp_alpha.sum(), 2))
