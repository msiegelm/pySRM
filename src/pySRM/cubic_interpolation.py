import numpy as np
from numba import jit_module

""" 2D """

def locate1d(xv, yv, xq, mm=2):
    """ Find left bracketing index for x with bisection

    input
    -----
    xv: array - x grid
    yv: array - y values
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


def cuint2(grid, f, fx, fy, fxy, query):
    """ 2D cubic interpolation for f: cuint2() -> f, fx, fy
    input
    -----
        grid: tuple - (xv: 1d array, yv: 1d array) - grid points
        f: 2d array - function values at grid points
        fx: 2d array - function derivative in x at grid points
        fy: 2d array - function derivative in y at grid points
        fxy: 2d array - function mixed derivative at grid points
        query: tuple - (xq: float, yq: float) - query point
    """

    # coefficients
    A1 = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [-3., 3., -2., -1.], [2., -2., 1., 1.]])
    A2 = np.array([[1., 0., -3., 2.], [0., 0., 3., -2.], [0., 1., -2., 1.], [0., 0., -1., 1.]])

    # unpack input
    xv, yv = grid
    xq, yq = query

    # find bracketing indices
    i = locate1d(xv, xv, xq)
    j = locate1d(yv, yv, yq)

    # define vertex of hypercube
    x1, x2 = xv[i], xv[i+1]
    y1, y2 = yv[j], yv[j+1]
    dx, dy = x2 - x1, y2 - y1
    xbar = (xq - x1) / dx
    ybar = (yq - y1) / dy

    # values at hypercube vertices
    ff = np.zeros((2, 2))
    ffx = np.zeros((2, 2))
    ffy = np.zeros((2, 2))
    ffxy = np.zeros((2, 2))
    for ii in range(2):
        for jj in range(2):
            ff[ii, jj] = f[i+ii, j+jj]
            ffx[ii, jj] = fx[i+ii, j+jj] * dx
            ffy[ii, jj] = fy[i+ii, j+jj] * dy
            ffxy[ii, jj] = fxy[i+ii, j+jj] * dx * dy
    Ftop = np.concatenate((ff, ffy), axis=1)
    Fbot = np.concatenate((ffx, ffxy), axis=1)
    F = np.concatenate((Ftop, Fbot), axis=0)

    # get coefficients
    alpha = A1 @ F @ A2

    # interpolate
    p, px, py = 0.0, 0.0, 0.0
    for ii in range(4):
        for jj in range(4):
            p = p + alpha[ii, jj] * xbar**ii * ybar**jj

    for ii in range(1, 4):
        for jj in range(4):
            px = px + alpha[ii, jj] * ii * xbar**(ii-1) * ybar**jj / dx

    for ii in range(4):
        for jj in range(1, 4):
            py = py + alpha[ii, jj] * xbar**ii * jj * ybar**(jj-1) / dy
    
    return p, px, py


def cuint2_currents(grid, Ugrid, Jgrid, query):
    """ 2D cubic interpolation for currents -> U=[u, v], J=[[dudx, dudy], [dvdx, dvdy]]
    input
    -----
        grid: tuple - (xv: 1d array, yv: 1d array) - grid points
        Ugrid: 3d array - [u, v] - current values at grid points (each component is 2d array)
        Jgrid: 4d array - [[dudx, dudy, dudxdy], [dvdx, dvdy, dvdxdy]] - current derivatives at grid points (each component is 2d array)
        query: tuple - (xq: float, yq: float) - query point
    output
    ------
        U_val: 1d array - [u_val, v_val] - interpolated current values at query point
        J_val: 2d array - [[dudx_val, dudy_val], [dvdx_val, dvdy_val]] - interpolated current derivatives at query point
    """
    u = Ugrid[0]
    dudx = Jgrid[0, 0]
    dudy = Jgrid[0, 1]
    dudxdy = Jgrid[0, 2]
    u_val, dudx_val, dudy_val = cuint2(grid, u, dudx, dudy, dudxdy, query)
    v = Ugrid[1]
    dvdx = Jgrid[1, 0]
    dvdy = Jgrid[1, 1]
    dvdxdy = Jgrid[1, 2]
    v_val, dvdx_val, dvdy_val = cuint2(grid, v, dvdx, dvdy, dvdxdy, query)

    U_val = np.array([u_val, v_val])
    J_val = np.array([[dudx_val, dudy_val], [dvdx_val, dvdy_val]])
    return U_val, J_val


jit_module(nopython=True, cache=True)
