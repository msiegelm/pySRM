import numpy as np
from numba import jit_module


def locate1d(xv, xq, mm=2):
    """ Find left bracketing index for x with bisection

    input
    -----
    xv: array - x grid
    xq: float - query point

    output
    ------
    jl : int - left bracketing index
    """

    n = len(xv)
    if n < 2 or mm < 2 or mm > n:
        raise ValueError("locate size error")

    ascnd = xv[n-1] >= xv[0]
    jl, ju = 0, n-1
    while ju - jl > 1:
        jm = (ju + jl) >> 1
        if (xq >= xv[jm]) == ascnd:
            jl = jm
        else:
            ju = jm

    return max(0, min(n - mm, jl - ((mm - 2) >> 1)))


def interp2d(grid, values, query):
    x1v, x2v = grid
    x1q, x2q = query

    # find bracketing indices
    i = locate1d(x1v, x1q)
    j = locate1d(x2v, x2q)

    # define hypercube vertex points
    xl1, xh1 = x1v[i], x1v[i+1]
    xl2, xh2 = x2v[j], x2v[j+1]

    # values at hypercube vertices
    C = np.zeros((2, 2))
    for ii in range(2):
        for jj in range(2):
            C[ii, jj] = values[i+ii, j+jj]

    # location of query point in hypercube
    xd1 = (x1q - xl1) / (xh1 - xl1)
    xd2 = (x2q - xl2) / (xh2 - xl2)

    # push through 2d hypercube
    C = push_hypercube(xd1, C)
    C = push_hypercube(xd2, C)
    return C


def interp2d_vec(grid, values, query):
    x1q, x2q = query
    q1f, q2f = x1q.flatten(), x2q.flatten()

    YQf = np.zeros(q1f.size)
    for i in range(q1f.size):
        YQf[i] = interp2d(grid, values, (q1f[i], q2f[i]))
    return YQf.reshape(x1q.shape)


def interp2dn(grid, values, query, n):
    x1v, x2v = grid
    x1q, x2q = query

    # check if query is outside grid
    if x1q < x1v[0] or x1q > x1v[-1] or x2q < x2v[0] or x2q > x2v[-1]:
        return np.nan*np.ones(n)
        # # top half on right side
        # if x2q > 0 and x1q > 0:
        #     return 1*np.ones(n)
        # else:
        #     return 4000*np.ones(n)
        # print('Query point outside grid')
        # return 1000*np.ones(n)

    # 

    # find bracketing indices
    i = locate1d(x1v, x1q)
    j = locate1d(x2v, x2q)

    # define hypercube vertex points
    xl1, xh1 = x1v[i], x1v[i+1]
    xl2, xh2 = x2v[j], x2v[j+1]

    # values at hypercube vertices
    C = np.zeros((2, 2, n))
    for ii in range(2):
        for jj in range(2):
            C[ii, jj, :] = values[i+ii, j+jj, :]

    # location of query point in hypercube
    xd1 = (x1q - xl1) / (xh1 - xl1)
    xd2 = (x2q - xl2) / (xh2 - xl2)

    # push through 2d hypercube
    C = push_hypercube(xd1, C)
    C = push_hypercube(xd2, C)
    return C


def push_hypercube(xid, C):
    """ Linear interpolation along 1st dimension of hypercube

    input
    -----
    xid : float - location of query point inside hyper cube (along 1st dimension)
    C : array_like - values at hyper cube vertices

    output
    ------
    C : array_like - values at hyper cube vertices decreased by 1 dimension
    """
    return C[0]*(1 - xid) + C[1]*xid


jit_module(nopython=True, cache=True)
