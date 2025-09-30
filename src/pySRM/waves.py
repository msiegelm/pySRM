import numpy as np
from numba import jit_module
from dispersion_newt import newt, init_newt

def calc_cg(k, h):
    g = 9.81
    c = np.sqrt(g/k * np.tanh(k*h))
    n = .5 * (1 + (2*k*h)/(np.sinh(2*k*h)))
    cg = c * n
    return cg


def calc_domega_dh(k, h):
    g = 9.81
    omega = np.sqrt(g*k*np.tanh(k*h))
    domega_dh = .5 * k * omega * 1/np.cosh(k*h) * 1/np.sinh(k*h)
    return domega_dh

def init_k_eckart(T, h):
    """ Calculate k using Eckart's Approximation 1951
    omega^2 = gk * sqrt(tanh( omega^2*h/g))
    """
    g = 9.81
    omega = 2*np.pi/T
    a = omega**2/g
    a0 = a*h
    k = a/np.sqrt(np.tanh(a0))
    return k


epsfcn, tolf, tolmin, STPMAX, maxiter = init_newt()

def init_k_iter(T, h):
    """ Calculate k using iterative method
    use ekart's approximation to get a first guess
    """

    g = 9.81

    # first guess
    kguess = init_k_eckart(T, h)
    omega = 2*np.pi/T
    k, _, _ = newt((h, omega), np.array([kguess]), tolf, tolmin, STPMAX, maxiter, epsfcn, verbose=True)
    return k[0]


jit_module(nopython=True, cache=True)
