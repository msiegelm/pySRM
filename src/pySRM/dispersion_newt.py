""" 
Description
-----------
Jit compiled Newton-Raphson solver for root finding of an n-dimensional function.

Date
----
Jan 2024

Author
------
Ray Young

Institution
-----------
SIO, UC San Diego

References
----------
Numerical Recipes, 3rd Edition, Section 9.6
"""

import numpy as np
from numba import jit_module
from numpy import abs, sum


def fdjac(params, x, fvec, epsfcn):
    """
    Description
    -----------
    Computes the Jacobian matrix of a function func evaluated at x.

    Parameters
    ----------
    func : function
        Function to be evaluated. func(x, params) must take in a vector x and return a vector fvec.
    params : tuple
        Parameters to be passed to func.
    x : ndarray
        Vector at which to evaluate the Jacobian.
    fvec : ndarray
        Vector of function values at x.
    eps : float
        Step size for finite difference approximation.

    Returns
    -------
    fjac : ndarray
        Jacobian matrix of func evaluated at x.

    References
    ----------
    Numerical Recipes, 3rd Edition, Section 9.7
    """
    n = len(x)
    fjac = np.zeros((n, n))
    xh = x.copy()
    for j in range(n):
        temp = x[j]
        h = epsfcn * abs(temp)
        if h == 0.0:
            h = epsfcn
        xh[j] = temp + h
        h = xh[j] - temp
        f = func(xh, params)
        xh[j] = temp
        for i in range(n):
            fjac[i, j] = (f[i] - fvec[i]) / h
    return fjac


def lnsrch(params, xold, fold, g, p, stpmax):
    """
    Description
    -----------
    Line search for the Newton-Raphson solver.

    Parameters
    ----------
    func : function 
        Function to be evaluated. function(x, params) must take in a vector x and return a vector fvec.
    params : tuple
        Parameters to be passed to func.
    xold : ndarray
        Old value of x.
    fold : float
        Value of func at xold.
    g : ndarray
        Gradient of func at xold.
    p : ndarray
        Step direction.
    stpmax : float
        Maximum step size.

    Returns
    -------
    x : ndarray
        New value of x.
    f : float
        Value of function to minimize at x.
    fvec : ndarray
        Vector of function values to be zeroed at x.
    check : bool
        False if normal exit. True if convergence on delta x.

    References
    ----------
    Numerical Recipes, 3rd Edition, Section 9.7
    """
    ALF = 1.0e-4
    TOLX = 1.0e-16

    n = len(xold)

    # rescale if the step is too big
    pnorm = np.sqrt(sum(p**2))
    if pnorm > stpmax:
        p *= stpmax / pnorm

    # compute slope
    slope = sum(g * p)
    if slope >= 0.0:
        raise ValueError('Roundoff problem in lnsrch.')

    test = 0.0
    for i in range(n):
        test = max(test, abs(p[i]) / max(abs(xold[i]), 1.0))

    alamin = TOLX / test
    alam = 1.0  # always try full Newton step first
    alam2 = 0.0
    f2 = 0.0

    # main loop
    while True:
        x = xold + alam * p  # new point
        f, fvec = fmin(params, x)  # evaluate function

        # convergence on delta x.
        if alam < alamin:
            x = xold
            check = True
            return x, f, fvec, check
        # sufficient function decrease
        elif f <= fold + ALF * alam * slope:
            check = False
            return x, f, fvec, check
        # backtrack
        else:
            # first time
            if alam == 1.0:
                tmplam = -slope / (2.0 * (f - fold - slope))
            # subsequent backtracks
            else:
                rhs1 = f - fold - alam * slope
                rhs2 = f2 - fold - alam2 * slope
                a = (rhs1 / (alam**2) - rhs2 / (alam2**2)) / (alam - alam2)
                b = (-alam2 * rhs1 / (alam**2) + alam * rhs2 / (alam2**2)) / (alam - alam2)
                if a == 0.0:
                    tmplam = -slope / (2.0 * b)
                else:
                    disc = b**2 - 3.0 * a * slope
                    if disc < 0.0:
                        tmplam = 0.5 * alam
                    elif b <= 0.0:
                        tmplam = (-b + np.sqrt(disc)) / (3.0 * a)
                    else:
                        tmplam = -slope / (b + np.sqrt(disc))
            if tmplam > 0.5 * alam:
                tmplam = 0.5 * alam
            alam2 = alam
            f2 = f
            alam = max(tmplam, 0.1 * alam)


def fmin(params, x):
    """
    Description
    -----------
    Returns the value of the function to be minimized and the function values at x.
    
    Parameters
    ----------
    func : function 
        Function to be evaluated. func(x, params) must take in a vector x and return a vector fvec.
    params : tuple
        Parameters to be passed to func.
    x : ndarray
        Vector at which to evaluate the function.

    Returns
    -------
    f : float
        Value of the function to be minimized at x.
    fvec : ndarray
        Vector of function values at x.
    """
    fvec = func(x, params)
    return 0.5 * sum(fvec**2), fvec


def newt(params, x, tolf, tolmin, STPMAX, maxiter, epsfcn, verbose=False):
    """
    Description
    -----------
    Jit compiled Newton-Raphson solver for root finding of an n-dimensional function with backtracking line search.

    Parameters
    ----------
    func : function
        Function to be evaluated. func(x, params) must take in a vector x and return a vector fvec to be zeroed.
    params : tuple
        Parameters to be passed to func.
    x : ndarray
        Initial guess for the root.
    tolf : float
        Tolerance for convergence of zero function.
    tolmin : float
        Tolerance for convergence of gradient.
    STPMAX : float
        Maximum step size for line search.
    maxiter : int
        Maximum number of iterations.
    epsfcn : float
        Step size for finite difference approximation.
    verbose : bool
        Print iteration information if True.

    Returns
    -------
    x : ndarray
        Root of func.
    passed : bool
        True if convergence criteria met, False otherwise.
    check: bool
        False if normal exit. True if spurious convergence.
    """

    TOLX = 1.0e-16  # tolerance for convergence of x (numeric limit)

    # initialize
    n = len(x)
    g = np.zeros(n)  # gradient
    p = np.zeros(n)  # step direction
    f, fvec = fmin(params, x)
    passed = True
    check = False
    # test initital guess for convergence. more strict than tolf
    test = 0.0
    for i in range(n):
        test = max(test, abs(fvec[i]))
    if test < 0.01 * tolf:
        check = False
        return x, passed, check

    # initialize stpmax for line search
    xmag = np.sqrt(sum(x**2))
    stpmax = STPMAX * max(xmag, n*1.0)

    # main loop
    for iter in range(maxiter):
        fjac = fdjac(params, x, fvec, epsfcn=epsfcn)  # compute Jacobian

        # compute gradient
        for i in range(n):
            temp = 0.0
            for j in range(n):
                temp += fjac[j, i] * fvec[j]
            g[i] = temp

        xold = x.copy()  # store x
        fold = f  # store f
        p = np.linalg.solve(fjac, -fvec)  # solve for step direction

        # line search
        x, f, fvec, check = lnsrch(params, xold, fold, g, p, stpmax)

        # test for convergence
        test = max(abs(fvec))
        if test < tolf:
            check = False
            if verbose:
                print('Function converged to within tolf')
                print('Iteration: ', iter,
                      'IC', x,
                      'Error: ', fvec,
                      '|Error|: ', f)
            return x, passed, check

        # test for convergence on grad f
        if check:
            den = max(f, 0.5 * n)
            test = 0.0
            for i in range(n):
                test = max(abs(g[i]) * max(abs(x[i]), 1.0) / den, test)
            check = test < tolmin
            passed = False
            if verbose:
                print('Gradient converged to within tolmin')
                print('Iteration: ', iter,
                      'IC', x,
                      'Error: ', fvec,
                      '|Error|: ', f)
            return x, passed, check

        # test for convergence on delta x
        test = 0.0
        for i in range(n):
            temp = abs(x[i] - xold[i]) / max(abs(x[i]), 1.0)
            test = max(test, temp)
        if test < TOLX:
            passed = False
            if verbose:
                print('Delta x converged to within TOLX')
                print('Iteration: ', iter,
                      'IC', x,
                      'Error: ', fvec,
                      '|Error|: ', f)
            return x, passed, check

        if verbose:
            print('Iteration: ', iter,
                  'IC', xold,
                  'ICn', x,
                  'Error: ', fvec,
                  '|Error|: ', f)

    passed = False
    return x, passed, check


def mnewt(params, x, tolf=1.0e-8, tolx=1.0e-8, maxiter=200, epsfcn=1.0e-8, verbose=False):
    """
    Description
    -----------
    Jit compiled Newton-Raphson solver for root finding of an n-dimensional function.

    Parameters
    ----------
    func : function
        Function to be evaluated. func(x, params) must take in a vector x and return a vector fvec.
    params : tuple
        Parameters to be passed to func.
    x : ndarray
        Initial guess for the root.
    fvec : ndarray
        Vector of function values at x.
    tolf : float
        Tolerance for convergence of zero function.
    tolx : float
        Tolerance for convergence of x.
    maxiter : int
        Maximum number of iterations.
    eps : float
        Step size for finite difference approximation.

    Returns
    -------
    x : ndarray
        Root of func.
    passed : bool
        True if convergence criteria met, False otherwise.
    """
    n = len(x)
    for iter in range(maxiter):
        fvec = func(x, params)
        fjac = fdjac(params, x, fvec, epsfcn)
        errf = sum(abs(fvec))
        if errf <= tolf:
            passed = True
            if verbose:
                print('Iteration: ', iter,
                      'IC', x,
                      'Error: ', fvec,
                      '|Error|: ', errf)
            return x, passed
        p = np.linalg.solve(fjac, -fvec)
        x = x + p
        errx = sum(abs(p))
        if errx <= tolx:
            passed = True
            if verbose:
                print('Iteration: ', iter,
                      'IC', x,
                      'Error: ', fvec,
                      '|Error|: ', errf)
            return x, passed

        if verbose:
            print('Iteration: ', iter,
                  'IC', x - p,
                  'ICn', x,
                  'Error: ', fvec,
                  '|Error|: ', errf)
    passed = False
    return x, passed


""" Test 2d function """


def func(x, params):
    g = 9.81
    h, omega = params
    k = x[0]
    f = np.sqrt(g*k*np.tanh(k*h)) - omega
    return np.array([f])


jit_module(nopython=True, cache=True)


def init_newt(epsfcn=1.0e-8, tolf=1.0e-8, tolmin=1.0e-12, STPMAX=100.0, maxiter=200):
    """ Initialize newt solver parameters based on NR

    Parameters
    ----------
    epsfcn : float
        Step size for finite difference approximation.
    tolf : float
        Tolerance for convergence of zero function.
    tolmin : float
        Tolerance for convergence of gradient.
    STPMAX : float
        Maximum step size for line search.
    maxiter : int
        Maximum number of iterations.
    """
    newt_params = (epsfcn, tolf, tolmin, STPMAX, maxiter)
    return newt_params
