""" Spectral wave refraction model for Palau

Description
------------

Date
----

Author
-------
RY

Institution
-----------
UCSD, SIO

References
-----------

Notes
-----

"""

# Basic imports
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from numba import njit
import netCDF4 as nc
import glob
import scipy.integrate as sc
import scipy.stats as ss
import glob
# from ms_toolbox import tools as tls
plt.rcParams["font.size"] = 16

# Custom imports
cpath = "/Users/miks/Desktop/Palau/Wave_Gauges/ray_tracer/ray"
# cpath = os.path.dirname(os.path.abspath(fpath))
sys.path.append(cpath + '/tools/')
from sgw_odeint import rk4, dp45, init_dp45
from interp import interp2dn, interp2d_vec
from dispersion_newt import newt, init_newt
from vicenty import inverse_vec
from waves import calc_cg
from transformation_matrix import *


""" Physical constants """
g = 9.81
""" Solver  Functions """

def default_dirfreq(freq=None,deg=None,degbins=None):
    """
    Returns CDIP frequency bins and 1deg directional bins
    """
    if not freq:
        freq = np.array([0.04  , 0.045 , 0.05  , 0.055 , 0.06  , 0.065 , 0.07  , 0.075 ,
        0.0813, 0.09  , 0.1   , 0.11  , 0.1225, 0.14  , 0.16  , 0.18  ,
        0.2075, 0.25  , 0.3125, 0.4   ])

    if not deg:
        deg = np.arange(1,361)
    if not degbins:
        degbins = np.arange(0,361)

    return freq,deg,degbins

def init_numerics():
    newt_params = init_newt()
    dp45_params_dense = init_dp45(kmax=100)
    dp45_params_final = init_dp45()

    return newt_params, dp45_params_dense, dp45_params_final

def init_bathy(x, y, z):
    """ Initialize bathymetry
    Pack bathymetry and its gradients into a 3d array [x, y, [z, dzdx, dzdy]]

    """
    # calculate gradient
    dzdx, dzdy = np.gradient(z, x, y)

    # setup for ray tracing
    Z = np.stack((z, dzdx, dzdy), axis=-1)
    grid = (x, y)
    return grid, Z

@njit(cache=True)
def init_K(h0, f0, newt_params):
    """ Initialize scalar wave number

    input
    -----
    h0 : float - depth at x0, y0
    f0 : float - frequency [hz]
    newt_params : tuple - newton solver parameters
    """

    # unpack newt params
    epsfcn, tolf, tolmin, STPMAX, maxiter = newt_params

    # calculate initial wave number for a given frequency
    om0 = 2*np.pi*f0
    kdw_guess = om0**2/g
    K0, _, _ = newt((h0, om0), np.array([kdw_guess]), tolf, tolmin, STPMAX, maxiter, epsfcn, verbose=False)
    return K0[0]


@njit(cache=True)
def single_ray(swr_params, th0):
    """ single ray tracing

    input
    -----
    swr_params : tuple - (grid, Z, x0, y0, K0, tf, dt, newt_params)
    th0 : float - angle of incidence

    output
    ------
    tout : array - time
    Xout : array - [x, y, kx, ky]
    """

    # unpack parameters
    grid, Z, x0, y0, K0, tf, dt, kh_crit, hmin, hmax, newt_params = swr_params
    ode_params = (grid, Z, hmin, hmax, kh_crit)

    kx0 = K0*np.cos(th0)
    ky0 = K0*np.sin(th0)
    Xstart = np.array([x0, y0, kx0, ky0])

    # ray trace
    tout, Xout, flag = rk4(0.0, tf, Xstart, dt, ode_params)
    return tout, Xout, flag


@njit(cache=True)
def calc_scaling_single(swr_params, xout):
    # unpack parameters
    grid, Z, x0, y0, K0, tf, dt, kh_crit, hmin, hmax, newt_params = swr_params

    # find last index
    idx = np.where(~np.isnan(xout[:, 0]))[0][-1]

    # calculate wave number
    K = np.sqrt(xout[0, 2]**2 + xout[0, 3]**2)  # near shore
    K0 = np.sqrt(xout[idx, 2]**2 + xout[idx, 3]**2)  # off shore

    # interp depth
    h = interp2dn(grid, Z, (xout[0, 0], xout[0, 1]), 3)[0]
    h0 = interp2dn(grid, Z, (xout[idx, 0], xout[idx, 1]), 3)[0]

    # calculate cg
    cg = calc_cg(K, h)
    cg0 = calc_cg(K0, h0)

    # calc final angle
    th0 = np.arctan2(xout[idx, 3], xout[idx, 2])

    S = K*cg0/K0/cg
    Kr = K/K0 # refraction
    Cgr = cg0/cg # shoaling 
    return S, th0, Kr, Cgr

# @njit(cache=True)
# def backward_ray_trace(tp0,x0,y0,grd,kh_crit=100,hmin=8,hmax=4000.0,saverays=True,dthparam=.01,dthOmax=1,th0=-np.pi/2,direction=1):
#     """
#     Send rays backwards from (x0,y0)

#     Input:
#     -----
#     tp0 = wave period  [float]
#     x0  = x position of sheltered point 
#     y0  = y position of sheltered point
#     grd = bathy grid (dictionary with variables x, y, z) x,y in UTM
#     kh_crit = critical depth wavenumber criteria #### RY is this being used?
#     hmin = minimum depth (defines hitting land)
#     hmax = maximum depth to run rays
#     dth = initial delta theta to spray rays
#     dth0_max = maximum adelta theta between offshore rays  
#     th_min_i = minimum delta theta onshore in degrees, if [], it changes with time 

#     Output:
#     ------
#     XXOUT   = ray position and wave number [x, y, kx, ky]
#     TTHETA  = sheltered ray direction (math,radians)
#     TTHETAO = offshore ray direction (math,radians)
#     FLAG    = indicates the ray is next to land
#     SS      = [k/ko]*[Cgo/Cg] scaling factor
#     XOUT_R1 = First ray
#     XOUT_R2 = Final ray

#     """
#     # Initialize Problem #
#     # ------------------ #
#     # print('Initializing Problem')

#     # numerical constants
#     newt_params, dp45_params_dense, dp45_params_final = init_numerics()

#     x = grd["x"]
#     y = grd["y"]
#     z = grd["z"]

#     # initialize bathymetry/calculate gradients
#     grid, Z = init_bathy(x, y, z)

#     # Initialize Ray Tracing #
#     # ---------------------- #
#     tf = -10*60*60  # maximum time for ray tracing
#     dt = -1  # time step for rk4 [s]
#     h0 = interp2dn(grid, Z, (x0, y0), 3)[0]

#     f0 = 1/tp0
#     K0 = init_K(h0, f0, newt_params)  # initial wave number
#     Kh0 = K0*h0
#     print('h0: ', h0)
#     print('K0: ', K0)
#     print('Kh0: ', Kh0)

#     # pack parameters
#     swr_params = (grid, Z, x0, y0, K0, tf, dt, kh_crit, hmin, hmax, newt_params)
#     tout, xout, flag, SS, thetas, thetaOs, Kratio, Cgratio = ray360(swr_params, direction=direction, th0=th0,dthparam=dthparam,dthOmax=dthOmax,saverays=saverays)
  
#     return tout, xout, flag, SS, thetas, thetaOs, Kratio, Cgratio


def pack_parameters(x,y,z,f0,tf=-10*60*60,dt = -1,x0=0,y0=0,kh_crit=100,hmin=8,hmax=4000,th0 = -np.pi/2,direction = 1,dthparam=.01,dthOmax=2.5,dthminC=None,saverays=False):
    """
    Packs parameters for ray tracing.

    Input:
    -----
    bathy:
        x = x in meters (0,0 corresponds with nearshore point if x0,y0=(0,0))
        y = y in meters (0,0 corresponds with nearshore pointif x0,y0=(0,0))
        z = z in meters (elevation)
    tf = maximum time for ray tracing (seconds)
    dt = time step for rk4 [s] (negative for backward ray tracing)
    x0 = x coordinate of nearshore point
    y0 = y coordinate of nearshore point
    kh_crit = []
    hmin = depth cut off determine ray has been blocked by land
    hmax = depth cut off for identifying ray has reached the deep ocean
    th0 = direction of initial ray (Default is north)
    direction = direction of rotation, 1 = CCW -1 = CW
    dthparam = initial nearshore delta theta in degrees
    dthparam = maximum allowed overshore delta theta in degrees
    dthminC = minimum allowed nearshore delta theta in degrees
    saverays = If output should include all rays
    Return:
    ------
    swr_params = parameters for ray tracing
    """
    newt_params, dp45_params_dense, dp45_params_final = init_numerics()
    grid, Z = init_bathy(x, y, z)
    h0 = interp2dn(grid, Z, (x0, y0), 3)[0]

    K0 = init_K(h0, f0, newt_params)
    Kh0 = K0*h0
    swr_params = (grid, Z, x0, y0, K0, tf, dt, kh_crit, hmin, hmax, newt_params)
    ray_params = (th0,direction,dthparam,dthOmax,dthminC,saverays)
        
    print("Initial Parameters:")
    print('h0: ', h0)
    print('K0: ', K0)
    print('Kh0: ', Kh0)

    return swr_params,ray_params

@njit(cache=True)
def ray360(swr_params, ray_params):
    """
    Shoots rays from initial point, 360 degrees around point.

    Input:
    -----
    swr_params = parameters for bathy and single ray behavior
    ray_params = parameters for 360 degree coverage
    
    """
    grid, Z, x0, y0, K0, tf, dt, kh_crit, hmin, hmax, newt_params = swr_params
    th0,direction,dthparam,dthOmax,dthminC,saverays = ray_params

    dthOmax = np.deg2rad(dthOmax)
    print("Initialize: 360 degree coverage")
    # initialize
    th = th0
    TOUT, XOUT, FLAG = single_ray(swr_params, th)
    idx = np.where(~np.isnan(XOUT[:, 0]))[0][-1] + 1
    S, th0, Kr, Cgr = calc_scaling_single(swr_params, XOUT[:idx,:])
    rcount = 1
    tout = []
    xout = []
    flag = []
    thetas = [] #sheltered thetas
    thetaOs = [] #offshore thetas
    Kratio = []
    Cgratio = []
    SS = []
    tout.append(TOUT[:idx])

    if saverays:
       xout.append(XOUT[:idx,:])
    flag.append(FLAG)
    thetas.append(th)
    thetaOs.append(th0)
    Kratio.append(Kr)
    Cgratio.append(Cgr)
    SS.append(S)
    # nn = int(360/dth)
    dthparam = dthparam*np.pi/180 #convert degrees to radians
    thcount = 0
    while thcount < 360:
        th = th + direction*dthparam
        TOUT, XOUT, FLAG = single_ray(swr_params, th)
        idx = np.where(~np.isnan(XOUT[:, 0]))[0][-1] + 1
        S_new, th0_new, Kr, Cgr = calc_scaling_single(swr_params, XOUT[:idx,:])
        rcount = rcount + 1
            #False = Land
        if (flag[-1] == False) and (FLAG == False):
            # print("Hit Land!")
            ## case where both rays hit land
            tout.append(TOUT[:idx])
            if saverays:
                xout.append(XOUT[:idx,:])
            flag.append(FLAG)
            thetas.append(th)
            thetaOs[-1] = np.nan
            thetaOs.append(np.nan)
            Kratio[-1] = np.nan
            Kratio.append(np.nan)
            Cgratio[-1] = np.nan
            Cgratio.append(np.nan)
            SS[-1] = np.nan
            SS.append(np.nan)

        else:
            thOprev = thetaOs[-1]
            thprev = thetas[-1]
            dthO = np.abs(th0_new - thOprev)
            dth = np.abs(th - thprev)
            th_new = th
            if dthminC == None:
                dthmin = 0.01/S_new*np.pi/180
            else:
                dthmin = dthminC
            jj = 0
            while (dthO > dthOmax) and (dth > dthmin):
                ## keep biesecting while dthO is greater than thresh and dth is greater than thresh   
                # print("Enter Bisection")        
                th_new = 0.5*(thprev + th_new)  # bisect
                TOUT, XOUT, FLAG = single_ray(swr_params, th_new)
                rcount = rcount + 1
                idx = np.where(~np.isnan(XOUT[:, 0]))[0][-1] + 1
                S_new, th0_new, Kr, Cgr = calc_scaling_single(swr_params, XOUT[:idx,:])
                dthO = np.abs(th0_new - thOprev)
                dth = np.abs(th_new - thprev)
                jj = jj + 1
            # print("Exit Bisection:", jj)        
            th = th_new #note: if while loop isn't tripped, th == th_new and th0 == th0_new
            th0 = th0_new #note: if while loop isn't tripped, th == th_new and th0 == th0_new
            tout.append(TOUT[:idx])
            if saverays:
                xout.append(XOUT[:idx,:])
            flag.append(FLAG)
            thetas.append(th)
            thetaOs.append(th0)
            Kratio.append(Kr)
            Cgratio.append(Cgr)
            SS.append(S_new)
        thcount = thcount + np.abs((thetas[-1]-thetas[-2]))*(180/np.pi)
        # print(thcount)
    print("Completed: 360 degree coverage")
    print("Rays used:", rcount)
    if thcount >= 360:
        return tout[:-1], xout[:-1], flag[:-1], SS[:-1], thetas[:-1], thetaOs[:-1], Kratio[:-1], Cgratio[:-1]
    else:
        return tout, xout, flag, SS, thetas, thetaOs, Kratio, Cgratio
