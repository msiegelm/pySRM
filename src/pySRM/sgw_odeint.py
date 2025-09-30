import numpy as np
from numba import jit_module
from interp import interp2dn
from waves import calc_cg
from cubic_interpolation import cuint2

""" Stepsize Controller Functions """


def success(err, h, reject, errold):
    """ Evaluate succes of DP45 step and update step size

    inputs
    ------
    err : float - error estimate
    h : float - step size
    reject : bool - if previous step was rejected
    errold : float - previous error estimate

    outputs
    -------
    success : bool - if step was successful
    hnext : float - step size for next step after success
    h : float - updated step size to retry step
    reject : bool - if step was rejected
    errold : float - updated previous error estimate
    """

    """ Static Constants """
    BETA = 0.0
    ALPHA = 0.2 - BETA*0.75
    SAFETY = 0.9
    MINSCALE = 0.2
    MAXSCALE = 10.
    
    """ evaluate success of step """
    if err <=1.0:  # step succeeded
        if (err == 0.0):
            scale = MAXSCALE
        else:
            scale = SAFETY*(err**(-ALPHA))*(errold**BETA)
            if scale < MINSCALE:
                scale = MINSCALE
            if scale > MAXSCALE:
                scale = MAXSCALE
        if reject:  # if previous step was rejected, dont allow step to grow
            hnext = h*min(scale, 1.0)
        else:
            hnext = h*scale
        errold = max(err, 1.0e-4)
        reject = False
        return True, hnext, h, reject, errold
    else:  # step failed
        scale = max(SAFETY*(err**(-ALPHA)), MINSCALE)
        h *= scale
        reject = True
        return False, h, h, reject, errold


""" Stepper Functions """


def dy(h, x, y, dydx, params):
    """ Apply single step of Dormand-Prince 4(5) method

    inputs
    ------
    h : float - step size
    x : float - independent variable
    y : array - dependent variable at x
    dydx : array - derivative of y at x
    ode_sys : function - function to compute derivative

    parameters updated
    -------
    yout : array - updated state
    dydxnew : array - updated derivative
    yerr : float - error estimate
    """

    """ Dormand-Prince Coefficients """
    c2, c3, c4, c5, c6, c7 = 1./5., 3./10., 4./5., 8./9., 1., 1.
    a21 = 1./5.
    a31, a32 = 3./40., 9./40.
    a41, a42, a43 = 44./45., -56./15., 32./9.
    a51, a52, a53, a54 = 19372./6561., -25360./2187., 64448./6561., -212./729.
    a61, a62, a63, a64, a65 = 9017./3168., -355./33., 46732./5247., 49./176., -5103./18656.
    a71, a72, a73, a74, a75, a76 = 35./384., 0., 500./1113., 125./192., -2187./6784., 11./84.
    b1, b2, b3, b4, b5, b6, b7 = 35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.
    bs1, bs2, bs3, bs4, bs5, bs6, bs7 = 5179./57600., 0., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.
    e1, e2, e3, e4, e5, e6, e7 = b1 - bs1, b2 - bs2, b3 - bs3, b4 - bs4, b5 - bs5, b6 - bs6, b7 - bs7

    """ take step """
    k2 = odefun(x + h*c2, y + h*a21*dydx, params)
    k3 = odefun(x + h*c3, y + h*(a31*dydx + a32*k2), params)
    k4 = odefun(x + h*c4, y + h*(a41*dydx + a42*k2 + a43*k3), params)
    k5 = odefun(x + h*c5, y + h*(a51*dydx + a52*k2 + a53*k3 + a54*k4), params)
    k6 = odefun(x + h*c6, y + h*(a61*dydx + a62*k2 + a63*k3 + a64*k4 + a65*k5), params)

    """ compute update """
    yout = y + h*(a71*dydx + a73*k3 + a74*k4 + a75*k5 + a76*k6)  # a72 = 0
    dydxnew = odefun(x + h*c7, yout, params)

    """ Compute error estimate """
    yerr = h*(e1*dydx + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*dydxnew)  # e2 = 0

    return yout, yerr, dydxnew, k2, k3, k4, k5, k6


def prepare_dense(h, y, yout, dydx, dydxnew, k3, k4, k5, k6):
    """ Prepare for dense output

    inputs
    ------
    h : float - step size
    y : array - dependent variable at x
    yout : array - updated y at x + h
    dydx : array - derivative of y at x
    dydxnew : array - updated derivative at x + h
    k3, k4, k5, k6 : arrays - intermediate steps

    outputs
    -------
    rcont1,...,rcont5 : arrays - coefficients for dense output
    """

    """ Dense output constants """
    d1 = -12715105075.0 / 11282082432.0
    d3 = 87487479700.0 / 32700410799.0
    d4 = -10690763975.0 / 1880347072.0
    d5 = 701980252875.0 / 199316789632.0
    d6 = -1453857185.0 / 822651844.0
    d7 = 69997945.0 / 29380423.0

    """ Update dense output coefficients """
    ydiff = yout - y
    bspl = h*dydx - ydiff

    rcont1 = y
    rcont2 = ydiff
    rcont3 = bspl
    rcont4 = ydiff - h*dydxnew - bspl
    rcont5 = h*(d1*dydx + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*dydxnew)
    return rcont1, rcont2, rcont3, rcont4, rcont5


def dense_out(xold, h, x, rcont1, rcont2, rcont3, rcont4, rcont5):
    """ Evaluate interpolatin polynomial for dense output

    inputs
    ------
    xold : float - old independent variable
    h : float - step size
    x : float - value of independent variable to interpolate to
    rcont1,...,rcont5 : arrays - coefficients for dense output

    outputs
    -------
    yinterp : array - interpolated value of dependent variable(s)
    """

    s = (x - xold)/h
    s1 = 1.0 - s
    yinterp = rcont1 + s*(rcont2 + s1*(rcont3 + s*(rcont4 + s1*rcont5)))
    return yinterp


def error(n, y, yout, yerr, atol, rtol):
    """ Compute error estimate

    inputs
    ------
    n : int - number of dependent variables
    y : array - dependent variable at x
    yout : array - updated y at x + h
    yerr : array - error estimate
    atol : float - absolute tolerance
    rtol : float - relative tolerance

    outputs
    -------
    err : float - error estimate
    """

    err = 0.
    for i in range(n):
        sk = atol + rtol*max(abs(y[i]), abs(yout[i]))
        err += (yerr[i]/sk)**2
    return np.sqrt(err/n)


def step(htry, x, y, dydx, n, atol, rtol, reject, errold, kmax, params):
    """ Take a single step of DP45 method with adaptive stepsize

    inputs
    ------
    htry : float - initial step size
    x : float - independent variable
    y : array - dependent variable at x
    dydx : array - derivative of y at x
    n : int - number of dependent variables
    atol : float - absolute tolerance
    rtol : float - relative tolerance
    reject : bool - if previous step was rejected
    errold : float - previous error estimate
    dense : int flag - 1 if dense output is requested, 0 or <0 otherwise

    outputs
    -------
    x : float - updated independent variable
    y : array - updated dependent variable
    dydx : array - updated derivative
    xold : float - old independent variable
    hdid : float - step size used
    hnext : float - step size for next step
    reject : bool - if step was rejected
    errold : float - updated previous error estimate
    rcont1,...,rcont5 : arrays - coefficients for dense output
    """

    """ Static Constants """
    h = htry
    while True:
        """ take step """
        yout, yerr, dydxnew, k2, k3, k4, k5, k6 = dy(h, x, y, dydx, params)

        """ compute error estimate """
        err = error(n, y, yout, yerr, atol, rtol)

        """ evaluate success of step """
        succeeded, hnext, h, reject, errold = success(err, h, reject, errold)
        if succeeded:
            break
    
    """ prepare for dense output """
    # if kmax > 0:
    rcont1, rcont2, rcont3, rcont4, rcont5 = prepare_dense(h, y, yout, dydx, dydxnew, k3, k4, k5, k6)

    """ bookkeeping """
    dydx = dydxnew
    y = yout
    xold = x
    hdid = h
    x += h

    return x, y, dydx, xold, hdid, hnext, reject, errold, rcont1, rcont2, rcont3, rcont4, rcont5


""" Wrapper Functions """

def dp45(x1, x2, ystart, params, atol, rtol, h1, hmin, max_steps, kmax):
    """ Static Constants """
    n = len(ystart)  # number of dependent variables

    """ Initialize Output """
    if kmax > 0: 
        xsave = np.linspace(x1, x2, kmax+1)
        ysave = np.zeros((kmax+1, n))
    elif kmax == 0:
        xsave = np.zeros(1)
        ysave = np.zeros((1, n))
    elif kmax == -1:
        xsave = np.zeros(max_steps+1)
        ysave = np.zeros((max_steps+1, n))
    else:
        raise ValueError("Invalid kmax")
    count = 0

    """ Initialize Step """
    x = x1  # initial independent variable
    y = ystart.copy() # initial dependent variable
    dydx = odefun(x1, ystart, params)  # initial derivative
    h = h1*np.sign(x2 - x1)  # initial step size    

    """ Initialize Step Size Controller """
    reject = False
    errold = 1.0e-4
    

    """ save initial values """
    xsave[0] = x
    ysave[0, :] = y
    count += 1

    """ book keeping """
    nok = 0
    nbad = 0

    """ Main Loop """ 
    for nstp in range(max_steps):
        if (x + h - x2) * (x2 - x1) > 0.0:
            h = x2 - x  # don't overstep
        x, y, dydx, xold, hdid, hnext, reject, errold, rcont1, rcont2, rcont3, rcont4, rcont5 = step(h, x, y, dydx, n, atol, rtol, reject, errold, kmax, params)

        if hdid == h:
            nok += 1
        else:
            nbad += 1
        
        # check termination
        term, flag = terminate(x, y, params)
        if term:
            return xsave, ysave, nok, nbad, flag

        # save output
        if kmax > 0:
            while (x - xsave[count])*(x2 - x1) > 0.0:
                ysave[count, :] = dense_out(xold, hdid, xsave[count], rcont1, rcont2, rcont3, rcont4, rcont5)
                count += 1
        elif kmax == 0:
            ysave[0, :] = y
            xsave[0] = x
        elif kmax == -1:
            xsave[count] = x
            ysave[count, :] = y
            count += 1

        # are we done?
        if (x - x2) * (x2 - x1) >= 0.0:
            if kmax > 0:
                ysave[count, :] = y
                count += 1 
            elif kmax == 0:
                ysave[0, :] = y
                xsave[0] = x
            elif kmax == -1:
                xsave[count] = x
                ysave[count, :] = y
                xsave = xsave[:count]
                ysave = ysave[:count, :]
            return xsave, ysave, nok, nbad, flag
        if abs(hnext) <= hmin:
            raise ValueError("Step size too small")
        h = hnext

    raise ValueError("Too many steps, try again")


def rk4(x1, x2, ystart, dx, params):
    """ Static Constants """
    n = len(ystart)  # number of dependent variables

    """ Initialize Output """
    xsave = np.arange(x1, x2, dx)
    if xsave[-1] != x2:
        xsave = np.append(xsave, x2)
    ysave = np.nan*np.ones((len(xsave), n))

    """ save initial values """
    ysave[0, :] = ystart

    """ Main Loop """
    for i in range(len(xsave)-1):
        # check for early termination
        term, flag = terminate(xsave[i], ysave[i, :], params)
        if term:
            # print("Terminated")
            xsave[i] = np.nan
            ysave[i, :] = np.nan
            return xsave, ysave, flag

        h = xsave[i+1] - xsave[i]
        k1 = odefun(xsave[i], ysave[i, :], params)
        k2 = odefun(xsave[i] + 0.5*h, ysave[i, :] + 0.5*h*k1, params)
        k3 = odefun(xsave[i] + 0.5*h, ysave[i, :] + 0.5*h*k2, params)
        k4 = odefun(xsave[i] + h, ysave[i, :] + h*k3, params)
        ysave[i+1, :] = ysave[i, :] + h*(k1 + 2.*k2 + 2.*k3 + k4)/6.
        # print(i)
    return xsave, ysave, flag


def euler(x1, x2, ystart, dx, params):
    """ Static Constants """
    n = len(ystart)  # number of dependent variables

    """ Initialize Output """
    xsave = np.arange(x1, x2, dx)
    if xsave[-1] != x2:
        xsave = np.append(xsave, x2)
    ysave = np.zeros((len(xsave), n))

    """ save initial values """
    ysave[0, :] = ystart

    """ Main Loop """
    for i in range(len(xsave)-1):
        h = xsave[i+1] - xsave[i]
        k1 = odefun(xsave[i], ysave[i, :], params)
        ysave[i+1, :] = ysave[i, :] + h*k1
    return xsave, ysave


""" specific ODE function """
# THIS IS THE FUNCTION THAT NEEDS TO BE CHANGED FOR DIFFERENT PROBLEMS #
########################################################################


def terminate(t, X, params):
    """ Function to terminate integration

    inputs
    ------
    t : float - independent variable
        x = time
    X : array - dependent variable
        X = [x, y, kx, ky], (x, y) = position, (kx, ky) = wavenumber vector
    params : tuple - parameters
        params = (grid, H, Hmin, kh_crit)
        grid : tuple - grid vectors
        H : array - depth 2d array [grid[0] x grid[1]]
        Hmin : float - minimum depth
        kh_crit : float - critical depth wavenumber
    outputs
    -------
    terminate : bool - if integration should terminate
    """

    # unpack parameters
    grid, H, Hmin, Hmax, kh_crit = params

    # unpack state
    x = X[0]
    y = X[1]
    # kx = X[2]
    # ky = X[3]

    # query environment
    Hq = interp2dn(grid, H, (x, y), 3)
    # Hq = cuint2(grid, H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], (x, y))

    h = Hq[0]

    # check if depth is below minimum
    if h <= Hmin:
        # print("Depth below minimum")
        flag = 0
        return True, flag

    # check if depth is above maximum
    if h >= Hmax:
        # print("Depth above maximum")
        flag = 1
        return True, flag

    # check if outside domain
    if x < grid[0][0]*.98 or y < grid[1][0]*.98 or y > grid[1][-1]*.98:
        # print("Outside domain")
        flag = 1
        return True, flag
    # check if out side to the right
    ## MS
    if x > 0.98*grid[0][-1]:
        # print("Outside to the right")
        flag = 1
        return True, flag
    ## MS
    # compute wavenumber
    # K = np.sqrt(kx**2 + ky**2)

    # check if we are in DW
    # if K*h > kh_crit:
    #     return True

    return False, -1


def odefun(t, X, params):
    """ Function to compute the derivative of the state

    Surface Gravity Wave Equation

    inputs
    ------
    t : float - independent variable
        x = time
    X : array - dependent variable
        X = [x, y, kx, ky], (x, y) = position, (kx, ky) = wavenumber vector
    params : tuple - parameters
        params = (grid, DJ)
        grid : tuple - grid vectors
        DJ : array - depth and depth gradient 5d array [grid[0] x grid[1] x 1 x 1 x 1]
    outputs
    -------
    dydt : array - derivative of the state
    """

    # unpack parameters
    grid, H, Hmin, Hmax, kh_crit = params

    # unpack state
    x = X[0]
    y = X[1]
    kx = X[2]
    ky = X[3]

    # query environment
    Hq = interp2dn(grid, H, (x, y), 3)
    # Hq = cuint2(grid, H[:,:,0], H[:,:,1], H[:,:,2], H[:,:,3], (x, y))
    h = Hq[0]
    dhdx = Hq[1]
    dhdy = Hq[2]

    # compute wavenumber
    K = np.sqrt(kx**2 + ky**2)

    # compute group velocity
    theta = np.arctan2(ky, kx)
    g = 9.81
    cg = calc_cg(K, h)
    cgx = cg * np.cos(theta)
    cgy = cg * np.sin(theta)
    dsigdh = -(g*K**2 * (np.tanh(K*h)**2 - 1)) / (2 * (g*K * np.tanh(K*h))**(1/2))

    # compute derivatives
    dxdt = cgx
    dydt = cgy
    dkxdt = -dsigdh * dhdx
    dkydt = -dsigdh * dhdy
    return np.array([dxdt, dydt, dkxdt, dkydt])


########################################################################
########################################################################


jit_module(nopython=True, cache=True)


def init_dp45(atol=1.0e-6, rtol=1.0e-6, h1=1.0e-3, hmin=0.0, max_steps=10000, kmax=-1):
    dp45_params = (atol, rtol, h1, hmin, max_steps, kmax)
    return dp45_params
