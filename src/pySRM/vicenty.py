""" 
Description
-----------
This module contains the functions for distance and bearing calculations
using the Vicenty formulae on the WGS84 ellipsoid.

Date
----
Jan 2023

Author
------
Ray Young

Institution
-----------
SIO, UCSD

References
----------
https://en.wikipedia.org/wiki/Vincenty%27s_formulae
"""

import numpy as np
from numba import jit_module

# WG84 ellipsoid parameters
a = 6378137.0  # semi-major axis [m]
f = 1/298.257223563  # flattening
b = 6356752.314245  # semi-minor axis [m]


def inverse(lat1, lon1, lat2, lon2, eps=1e-12, max_iter=100):
    """
    Inverse calculation of distance and bearing between two points on the
    WGS84 ellipsoid using the Vicenty formulae.

    Parameters
    ----------
    lat1 : float
        Latitude of point 1 [degrees]
    lon1 : float
        Longitude of point 1 [degrees]
    lat2 : float
        Latitude of point 2 [degrees]
    lon2 : float
        Longitude of point 2 [degrees]
    eps : float, optional
        Convergence criterion (approximately 0.06 mm)
    Returns
    -------
    s : float
        Distance between points 1 and 2 [m]
    alpha1 : float
        Forward azimuth at point 1 [degrees]
    alpha2 : float
        Reverse azimuth at point 2 [degrees]

    Notes
    -----
    This is basically all from the Wikipedia page and copilot
    """

    # convert to radians
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    # reduced latitudes
    U1 = np.arctan((1-f)*np.tan(lat1))
    U2 = np.arctan((1-f)*np.tan(lat2))

    # initial values
    L = lon2 - lon1
    Lambda = L
    Lambda_p = np.inf
    count = 0
    # iterate until convergence
    while np.abs(Lambda-Lambda_p) > eps and count < max_iter:
        sin_sigma = np.sqrt((np.cos(U2)*np.sin(Lambda))**2 +
                            (np.cos(U1)*np.sin(U2) -
                             np.sin(U1)*np.cos(U2)*np.cos(Lambda))**2)
        cos_sigma = (np.sin(U1)*np.sin(U2) +
                     np.cos(U1)*np.cos(U2)*np.cos(Lambda))
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sin_alpha = (np.cos(U1)*np.cos(U2)*np.sin(Lambda) /
                     np.sin(sigma))
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*np.sin(U1)*np.sin(U2)/cos_sq_alpha
        C = f/16*cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))
        Lambda_p = Lambda
        Lambda = L + (1-C)*f*sin_alpha*(sigma + C*sin_sigma *
                                        (cos_2sigma_m + C*cos_sigma *
                                         (-1 + 2*cos_2sigma_m**2)))
        count += 1

    if count == max_iter:
        print('Warning: maximum number of iterations reached')

    # final values
    u_sq = cos_sq_alpha*(a**2 - b**2)/b**2
    A = 1 + u_sq/16384*(4096 + u_sq*(-768 + u_sq*(320 - 175*u_sq)))
    B = u_sq/1024*(256 + u_sq*(-128 + u_sq*(74 - 47*u_sq)))
    delta_sigma = B*sin_sigma*(cos_2sigma_m + B/4 *
                               (cos_sigma*(-1 + 2*cos_2sigma_m**2) -
                                B/6*cos_2sigma_m*(-3 + 4*sin_sigma**2)*
                                (-3 + 4*cos_2sigma_m**2)))
    s = b*A*(sigma - delta_sigma)
    alpha1 = np.rad2deg(np.arctan2(np.cos(U2)*np.sin(Lambda),
                                   np.cos(U1)*np.sin(U2) -
                                   np.sin(U1)*np.cos(U2)*np.cos(Lambda)))
    alpha2 = np.rad2deg(np.arctan2(np.cos(U1)*np.sin(Lambda),
                                   -np.sin(U1)*np.cos(U2) +
                                   np.cos(U1)*np.sin(U2)*np.cos(Lambda)))
    return s, alpha1, alpha2


def inverse_vec(lat1, lon1, lat2, lon2, eps=1e-12, max_iter=100):
    lat1f = lat1.flatten()
    lon1f = lon1.flatten()
    lat2f = lat2.flatten()
    lon2f = lon2.flatten()

    s = np.zeros(lat1f.shape)
    alpha1 = np.zeros(lat1f.shape)
    alpha2 = np.zeros(lat1f.shape)
    for i in range(lat1f.shape[0]):
        # check if (lat1, lon1) == (lat2, lon2)
        if lat1f[i] == lat2f[i] and lon1f[i] == lon2f[i]:
            s[i] = 0
            alpha1[i] = 0
            alpha2[i] = 0
        else:
            s[i], alpha1[i], alpha2[i] = inverse(
                    lat1f[i], lon1f[i],
                    lat2f[i], lon2f[i],
                    eps, max_iter)
    s = s.reshape(lat1.shape)
    alpha1 = alpha1.reshape(lat1.shape)
    alpha2 = alpha2.reshape(lat1.shape)
    return s, alpha1, alpha2


def direct(lat1, lon1, alpha1, s, eps=1e-12, max_iter=100):
    """
    Direct calculation of distance and bearing between two points on the
    WGS84 ellipsoid using the Vicenty formulae.

    Parameters
    ----------
    lat1 : float
        Latitude of point 1 [degrees]
    lon1 : float
        Longitude of point 1 [degrees]
    alpha1 : float
        Forward azimuth at point 1 [degrees]
    s : float
        Distance between points 1 and 2 [m]
    eps : float, optional
        Convergence criterion (approximately 0.06 mm)
    Returns
    -------
    lat2 : float
        Latitude of point 2 [degrees]
    lon2 : float
        Longitude of point 2 [degrees]
    alpha2 : float
        Reverse azimuth at point 2 [degrees]

    Notes
    -----
    This is basically all from the Wikipedia page and copilot
    """

    # convert to radians
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    alpha1 = np.deg2rad(alpha1)

    # reduced latitudes
    U1 = np.arctan((1-f)*np.tan(lat1))

    # initial values
    sigma1 = np.arctan2(np.tan(U1), np.cos(alpha1))
    sin_alpha = np.cos(U1)*np.sin(alpha1)
    cos_sq_alpha = 1 - sin_alpha**2
    u_sq = cos_sq_alpha*(a**2 - b**2)/b**2
    # k1 = (np.sqrt(1 + u_sq) - 1)/(np.sqrt(1 + u_sq) + 1)
    # A = (1 + 0.25*k1**2)/(1 - k1)
    # B = k1*(1 - 3/(8*k1**2))

    A = 1 + u_sq/16384*(4096 + u_sq*(-768 + u_sq*(320 - 175*u_sq)))
    B = u_sq/1024*(256 + u_sq*(-128 + u_sq*(74 - 47*u_sq)))

    # iterate until convergence
    sigma = s/(b*A)
    sigma_p = np.inf
    count = 0
    while np.abs(sigma - sigma_p) > eps and count < max_iter:
        _2sigma_m = 2.*sigma1 + sigma
        delta_sigma = B*np.sin(sigma)*( np.cos(_2sigma_m) + B/4. *
                                       (np.cos(sigma)*(-1 + 2*np.cos(_2sigma_m)**2) -
                                        B/6.*np.cos(_2sigma_m)*(-3 + 4*np.sin(sigma)**2)*
                                        (-3 + 4*np.cos(_2sigma_m)**2)))
        sigma_p = sigma
        sigma = s/(b*A) + delta_sigma
        count += 1

    if count == max_iter:
        print('Warning: maximum number of iterations reached')

    # final values
    sin_sigma = np.sin(sigma)
    cos_sigma = np.cos(sigma)
    lat2 = np.arctan2(np.sin(U1)*cos_sigma + np.cos(U1)*sin_sigma*np.cos(alpha1),
                      (1-f)*np.sqrt(sin_alpha**2 + (np.sin(U1)*sin_sigma -
                                                    np.cos(U1)*cos_sigma*np.cos(alpha1))**2))
    Lambda = np.arctan2(sin_sigma*np.sin(alpha1),
                        np.cos(U1)*cos_sigma - np.sin(U1)*sin_sigma*np.cos(alpha1))
    C = f/16*cos_sq_alpha*(4 + f*(4 - 3*cos_sq_alpha))
    L = Lambda - (1-C)*f*sin_alpha*(sigma + C*sin_sigma *
                                    (np.cos(_2sigma_m) + C*cos_sigma *
                                     (-1 + 2*np.cos(_2sigma_m)**2)))
    lon2 = lon1 + L
    alpha2 = np.rad2deg(np.arctan2(sin_alpha, -np.sin(U1)*sin_sigma +
                                   np.cos(U1)*cos_sigma*np.cos(alpha1)))
    return np.rad2deg(lat2), np.rad2deg(lon2), alpha2


jit_module(nopython=True, cache=True)
