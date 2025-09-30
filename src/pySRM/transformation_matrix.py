"""
Author
-----
MNS

"""

import numpy as np
import scipy.stats as ss
import netCDF4 as nc

def direction_math2naut(theta):
    thetanew = 270 - theta
    if type(theta) == float or type(theta) == int:
        if thetanew > 360:
            thetanew = thetanew - 360
        elif thetanew < 0:
            thetanew = 90 - thetanew
    else:
        kk = (thetanew > 360)
        thetanew[kk] = thetanew[kk] - 360
        kkz = (thetanew < 0)
        thetanew[kkz] = 90 - thetanew[kkz]

    return thetanew

# def get_E_ratio(TTHETA,TTHETAO,XOUT_R1,XOUT_R2,deg):
#     th_on = np.rad2deg(TTHETA) #math
#     th_off = np.rad2deg(TTHETAO) #math
#     tnau = direction_math2naut(th_on[:-1])
#     tnau0 = direction_math2naut(th_off[:-1])

#     #### add ending ray
#     # r1 = np.rad2deg(np.angle(XOUT_R1[:,2]+ 1j*XOUT_R1[:,3]))
#     r2 = np.rad2deg(np.angle(XOUT_R2[:,2]+ 1j*XOUT_R2[:,3]))

#     # kk1 = np.isnan(r1)
#     kk2 = np.isnan(r2)

#     # r1 = r1[~kk1]
#     r2 = r2[~kk2]

#     # r1s0 = np.array([r1[0],r1[-1]])
#     r2s0 = np.array([r2[0],r2[-1]])

#     # rnau1 = direction_math2naut(r1s0)
#     rnau2 = direction_math2naut(r2s0)

#     tnau = np.hstack((tnau,rnau2[0]))
#     tnau0 = np.hstack((tnau0,rnau2[-1]))

#     #bin bounds
#     degb = np.hstack([0,deg])

#     tsum = np.ones(len(degb)-1)*np.nan
#     tsum0 = np.ones(len(degb)-1)*np.nan
#     rar = np.ones(len(degb)-1)*np.nan

#     for jd in range(len(degb)-1):
#         d1 = degb[jd]
#         d2 = degb[jd+1]

#         kk = np.where((tnau0>=d1) & (tnau0 < d2))[0]
#         if np.any(kk):
#             dthetaoff = np.ones(len(kk))*np.nan
#             dthetaon = np.ones(len(kk))*np.nan
#             for jl in range(len(kk)):
#                 kkp1 = kk[jl] + 1
#                 if kkp1 < len(tnau0):
#                     dth = tnau0[kk[jl]] - tnau0[kkp1]
#                     if dth < 0:
#                         dthetaoff[jl] = np.nan
#                         dthetaon[jl] = np.nan

#                     else:
#                         dthetaoff[jl] =  tnau0[kk[jl]] - tnau0[kkp1]
#                         dthetaon[jl] = tnau[kk[jl]] - tnau[kkp1]
#             ras = np.nanmean(dthetaoff / dthetaon)

#             tsum0[jd] = np.nansum(np.abs(dthetaoff))
#             tsum[jd] = np.nansum(np.abs(dthetaon))
#             rar[jd] = ras
#     kkn = (tsum0 == 0)
#     tsum[kkn] = np.nan
#     ratio = tsum / tsum0 #on / off

#     kkn = np.isnan(ratio)
#     ratio[kkn] = 0
#     return ratio,deg,tsum,tsum0,rar


def get_theta(TTHETA,TTHETAO):
    """
    Get on and offshore wave direction in nautical direction
    Input:
    -----
    TTHETA = wave direction on shore (math and radians)
    TTHETAO = wave direction off shore (math and radians)

    Output:
    ------
    tnau = wave direction on shore (nautical and degrees)
    tnau0 = wave direction off shore (nautical and degrees)
    """
    th_on = np.rad2deg(TTHETA)
    th_off = np.rad2deg(TTHETAO)
    tnau = direction_math2naut(th_on)
    tnau0 = direction_math2naut(th_off)
    return tnau,tnau0

def get_ray_indices(TTHETAO,d1,d2):
    th_off = np.rad2deg(TTHETAO) #math
    tnau0 = direction_math2naut(th_off)
    kk = np.where((tnau0>=d1) & (tnau0 < d2))[0]
    return kk

def get_dtheta(tnau,tnau0,kk):
    dthetaoff = np.ones(len(kk))*np.nan
    dthetaon = np.ones(len(kk))*np.nan
    for jl in range(len(kk)):
        kkp1 = kk[jl] + 1
        if kkp1 < len(tnau0):
            dth = tnau0[kk[jl]] - tnau0[kkp1]
            if dth < 0:
                dthetaoff[jl] = np.nan
                dthetaon[jl] = np.nan

            else:
                dthetaoff[jl] =  tnau0[kk[jl]] - tnau0[kkp1]
                dthetaon[jl] = tnau[kk[jl]] - tnau[kkp1]
    return dthetaoff,dthetaon

def LI2degbin(degbins,tnau,tnau0,SS,Kratio=[],Cgratio=[]):
    """
    Linear interpolate between 2 points to land on degree bins.

    Input:
    -----
    degbins = degree bins (typically 0-360)
    tnau    = theta onshore
    tnau0   = theta offshore

    Return:
    -----
    tnau_new  = interpolated theta onshore
    tnau0_new = interpolated theta offshore
    """
    nt = len(tnau)

    tnau_new = np.array([])
    tnau0_new = np.array([])
    # SS_new = np.array([])
    for jt in range(nt-1):
        t1 = tnau0[jt]
        t2 = tnau0[jt+1]
        tnau_new = np.hstack((tnau_new,tnau[jt]))
        tnau0_new = np.hstack((tnau0_new,tnau0[jt]))
        # SS_new = np.hstack((SS_new,SS[jt]))
        if not np.isnan(t1) and not np.isnan(t2):
            # ignores transition across 0-360
            if ((360 - t1 < 90) and (t2 < 90)):
                ## puts points at discontinuity from 360 to 0
                t2cont = 360 + t2

                p1 = (tnau[jt],tnau0[jt])
                p2 = (tnau[jt+1],t2cont)
                xx = [p1[0],p2[0]]
                yy = [p1[1],p2[1]]
                lr = ss.linregress(xx,yy)
                XX = (360 - lr.intercept) / lr.slope
                tnau_new = np.hstack((tnau_new,XX))
                tnau0_new = np.hstack((tnau0_new,360))

                t1cont = t1 - 360
                p1 = (tnau[jt],t1cont)
                p2 = (tnau[jt+1],tnau0[jt+1])
                xx = [p1[0],p2[0]]
                yy = [p1[1],p2[1]]
                lr = ss.linregress(xx,yy)
                XX = (0 - lr.intercept) / lr.slope
                tnau_new = np.hstack((tnau_new,XX))
                tnau0_new = np.hstack((tnau0_new,0))
            elif ((t1 < 90) and (360-t2 < 90)):
                ## puts points at discontinuity from 0 to 360
                t2cont = t2 - 360
                p1 = (tnau[jt],tnau0[jt])
                p2 = (tnau[jt+1],t2cont)
                xx = [p1[0],p2[0]]
                yy = [p1[1],p2[1]]
                lr = ss.linregress(xx,yy)
                XX = (0 - lr.intercept) / lr.slope
                tnau_new = np.hstack((tnau_new,XX))
                tnau0_new = np.hstack((tnau0_new,0))

                t1cont = t1 + 360
                p1 = (tnau[jt],t1cont)
                p2 = (tnau[jt+1],tnau0[jt+1])
                xx = [p1[0],p2[0]]
                yy = [p1[1],p2[1]]
                lr = ss.linregress(xx,yy)
                XX = (360 - lr.intercept) / lr.slope
                tnau_new = np.hstack((tnau_new,XX))
                tnau0_new = np.hstack((tnau0_new,360))

            else:
                if t1 < t2:
                    kk = np.where((degbins > t1) & (degbins <t2))[0]
                else:
                    kk = np.where((degbins > t2) & (degbins <t1))[0]
                if np.any(kk):
                    p1 = (tnau[jt],tnau0[jt])
                    p2 = (tnau[jt+1],tnau0[jt+1])
                    xx = [p1[0],p2[0]]
                    yy = [p1[1],p2[1]]
                    lr = ss.linregress(xx,yy)
                    dbkk = degbins[kk]
                    XX = (dbkk - lr.intercept) / lr.slope
                    tnau_new = np.hstack((tnau_new,XX))
                    tnau0_new = np.hstack((tnau0_new,dbkk))
    nt = len(tnau_new)
    # fills between last point and inserted endpoint (either 0 or 360)
    tnau_new2 = np.array([])
    tnau0_new2 = np.array([])
    # SS_new = np.array([])
    for jt in range(nt-1):
        t1 = tnau0_new[jt]
        t2 = tnau0_new[jt+1]
        tnau_new2 = np.hstack((tnau_new2,tnau_new[jt]))
        tnau0_new2 = np.hstack((tnau0_new2,tnau0_new[jt]))

        if not np.abs(t1-t2) == 360:
            if t1 < t2:
                kk = np.where((degbins > t1) & (degbins <t2))[0]
            else:
                kk = np.where((degbins > t2) & (degbins <t1))[0]
            if np.any(kk):
                p1 = (tnau_new[jt],tnau0_new[jt])
                p2 = (tnau_new[jt+1],tnau0_new[jt+1])
                xx = [p1[0],p2[0]]
                yy = [p1[1],p2[1]]
                lr = ss.linregress(xx,yy)
                dbkk = degbins[kk]
                XX = (dbkk - lr.intercept) / lr.slope
                tnau_new2 = np.hstack((tnau_new2,XX))
                tnau0_new2 = np.hstack((tnau0_new2,dbkk))

    if tnau[0] > tnau[-1]:
        #makes sure tnau is in ascending order
        tnau = np.flipud(tnau)
        tnau_new2 = np.flipud(tnau_new2)
        tnau0_new2 = np.flipud(tnau0_new2)
        SS = np.flipud(SS)
        if np.any(Kratio):
            Kratio = np.flipud(Kratio)
        if np.any(Cgratio):
            Cgratio = np.flipud(Cgratio)
    SS_new = np.interp(tnau_new2,tnau,SS)
    if np.any(Kratio):
        Kratio_new = np.interp(tnau_new2,tnau,Kratio)
    if np.any(Cgratio):
        Cgratio_new = np.interp(tnau_new2,tnau,Cgratio)

    if np.any(Kratio) and not np.any(Cgratio):
        #only Kratio
        return tnau_new2, tnau0_new2, SS_new, Kratio_new
    elif np.any(Kratio) and np.any(Cgratio):
        #Kratio and Cgratio
        return tnau_new2, tnau0_new2, SS_new, Kratio_new, Cgratio_new
    elif not np.any(Kratio) and np.any(Cgratio):
        #only Cgratio
        return tnau_new2, tnau0_new2, SS_new, Cgratio_new
    elif not np.any(Kratio) and not np.any(Cgratio):
            #neither Kratio nor Cgratio
        return tnau_new2, tnau0_new2, SS_new


#
# def fill_islands(tnau,tnau0,SS,FLAG):
#     """
#     Inserts NaNs where land blocking would be.
#
#     """
#     dth = np.diff(tnau)
#
#     if np.nanmean(dth) < 0:
#         tnau = np.flipud(tnau)
#         tnau0 = np.flipud(tnau0)
#         SS = np.flipud(SS)
#         FLAG = np.flipud(FLAG)
#
#     dthmed = .01 #np.median(dth)
#
#     #### add end bounds ######
#     if tnau[0] > 0:
#         tnauadd = np.arange(0,tnau[0],dthmed)
#         tnau = np.hstack((tnauadd,tnau))
#         tnau0 = np.hstack((np.nan*np.ones(len(tnauadd)),tnau0))
#         SS = np.hstack((np.nan*np.ones(len(tnauadd)),SS))
#         FLAG = np.hstack((np.zeros(len(tnauadd)),FLAG))
#
#     if tnau[-1] < 360:
#         tnauadd = np.arange(tnau[-1]+dthmed,360+dthmed,dthmed)
#         tnau = np.hstack((tnau,tnauadd))
#         tnau0 = np.hstack((tnau0,np.nan*np.ones(len(tnauadd))))
#         SS = np.hstack((SS,np.nan*np.ones(len(tnauadd))))
#         FLAG = np.hstack((FLAG,np.zeros(len(tnauadd))))
#
#     #### identifies if there are two FLAGs in a row (meaning between these rays there is a land strike)
#     #### inserts nan between these rays
#     for jt in range(len(FLAG)):
#         flat = FLAG[jt]
#         if jt == 0:
#             tnau_new = []
#             tnau0_new = []
#             sca_new = []
#             fla_new = []
#
#         if flat == 0:
#             tnau_new.append(tnau[jt])
#             tnau0_new.append(tnau0[jt])
#             sca_new.append(SS[jt])
#             fla_new.append(0)
#         if (jt < (len(FLAG)-1)):
#
#             if (flat==1) and (FLAG[jt+1] == 1):
#                 tnau_new.append(tnau[jt])
#                 tnau0_new.append(tnau0[jt])
#                 sca_new.append(SS[jt])
#                 fla_new.append(FLAG[jt])
#
#                 tnau_mn = .5* (tnau[jt] + tnau[jt+1])
#                 tnau_new.append(tnau_mn)
#                 tnau0_new.append(np.nan)
#                 sca_new.append(np.nan)
#                 fla_new.append(1)
#             elif (flat == 1) and (FLAG[jt+1] == 0):
#                 tnau_new.append(tnau[jt])
#                 tnau0_new.append(tnau0[jt])
#                 sca_new.append(SS[jt])
#                 fla_new.append(0)
#     ### NaNs out single rays that slip between land
#     for jt in range(len(tnau_new)):
#         if jt == 0:
#             if tnau0_new[jt+1] == np.nan:
#                 tnau_new[jt] = np.nan
#                 sca_new[jt] = np.nan
#                 fla_new[jt] = 1
#         elif jt == len(tnau_new)-1:
#             if tnau0_new[jt-1] == np.nan:
#                 tnau_new[jt] = np.nan
#                 sca_new[jt] = np.nan
#                 fla_new[jt] = 1
#         else:
#             if (np.isnan(tnau0_new[jt-1])) and (np.isnan(tnau0_new[jt+1])):
#                 tnau0_new[jt] = np.nan
#                 sca_new[jt] = np.nan
#                 fla_new[jt] = 1
#
#
#     return tnau_new,tnau0_new,sca_new,fla_new



# def LI2degbin(degbins,tnau,tnau0,SS):
#     """
#     Linear interpolate between 2 points to land on degree bins.

#     Input:
#     -----
#     degbins = degree bins (typically 0-360)
#     tnau    = theta onshore
#     tnau0   = theta offshore

#     Return:
#     -----
#     tnau_new  = interpolated theta onshore
#     tnau0_new = interpolated theta offshore
#     """
#     nt = len(tnau)

#     tnau_new = np.array([])
#     tnau0_new = np.array([])
#     # SS_new = np.array([])
#     for jt in range(nt-1):
#         t1 = tnau0[jt]
#         t2 = tnau0[jt+1]
#         tnau_new = np.hstack((tnau_new,tnau[jt]))
#         tnau0_new = np.hstack((tnau0_new,tnau0[jt]))
#         # SS_new = np.hstack((SS_new,SS[jt]))
#         if not np.isnan(t1) and not np.isnan(t2):
#             # ignores transition across 0-360
#             if ((360-t1 < 1) and (t2 < 1)):
#                 ## puts points at discontinuity from 360 to 0
#                 t2cont = 360 + t2

#                 p1 = (tnau[jt],tnau0[jt])
#                 p2 = (tnau[jt+1],t2cont)
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (360 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,360))

#                 t1cont = t1 - 360
#                 p1 = (tnau[jt],t1cont)
#                 p2 = (tnau[jt+1],tnau0[jt+1])
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (0 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,0))
#             elif ((t1 < 1) and (360-t2 < 1)):
#                 ## puts points at discontinuity from 0 to 360
#                 t2cont = t2 - 360
#                 p1 = (tnau[jt],tnau0[jt])
#                 p2 = (tnau[jt+1],t2cont)
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (0 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,0))

#                 t1cont = t1 + 360
#                 p1 = (tnau[jt],t1cont)
#                 p2 = (tnau[jt+1],tnau0[jt+1])
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (360 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,360))

#             else:
#                 if t1 < t2:
#                     kk = np.where((degbins > t1) & (degbins <t2))[0]
#                 else:
#                     kk = np.where((degbins > t2) & (degbins <t1))[0]
#                 if np.any(kk):
#                     p1 = (tnau[jt],tnau0[jt])
#                     p2 = (tnau[jt+1],tnau0[jt+1])
#                     xx = [p1[0],p2[0]]
#                     yy = [p1[1],p2[1]]
#                     lr = ss.linregress(xx,yy)
#                     dbkk = degbins[kk]
#                     XX = (dbkk - lr.intercept) / lr.slope
#                     tnau_new = np.hstack((tnau_new,XX))
#                     tnau0_new = np.hstack((tnau0_new,dbkk))
#     SS_new = np.interp(tnau_new,tnau,SS)
#     # nt = len(tnau_new)
#     # for jt in range(nt-1):
#     #     p1ss = (tnau_new[jt],SS[jt])
#     #     p2ss = (tnau[jt+1],SS[jt+1])
#     #     xxss = [p1ss[0],p2ss[0]]
#     #     yyss = [p1ss[1],p2ss[1]]
#     #     lrss = ss.linregress(xxss,yyss)
#     #     XXss = (dbkk - lrss.intercept) / lrss.slope
#     # SS_new = np.hstack((SS_new,XXss))

#     return tnau_new, tnau0_new, SS_new

# def LI2degbin(degbins,tnau,tnau0,SS):
#     """
#     Linear interpolate between 2 points to land on degree bins.

#     Input:
#     -----
#     degbins = degree bins (typically 0-360)
#     tnau    = theta onshore
#     tnau0   = theta offshore

#     Return:
#     -----
#     tnau_new  = interpolated theta onshore
#     tnau0_new = interpolated theta offshore
#     """
#     nt = len(tnau)

#     tnau_new = np.array([])
#     tnau0_new = np.array([])
#     # SS_new = np.array([])
#     for jt in range(nt-1):
#         t1 = tnau0[jt]
#         t2 = tnau0[jt+1]
#         tnau_new = np.hstack((tnau_new,tnau[jt]))
#         tnau0_new = np.hstack((tnau0_new,tnau0[jt]))
#         # SS_new = np.hstack((SS_new,SS[jt]))
#         if not np.isnan(t1) and not np.isnan(t2):
#             # ignores transition across 0-360
#             if ((360 - t1 < 90) and (t2 < 90)):
#                 ## puts points at discontinuity from 360 to 0
#                 t2cont = 360 + t2

#                 p1 = (tnau[jt],tnau0[jt])
#                 p2 = (tnau[jt+1],t2cont)
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (360 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,360))

#                 t1cont = t1 - 360
#                 p1 = (tnau[jt],t1cont)
#                 p2 = (tnau[jt+1],tnau0[jt+1])
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (0 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,0))
#             elif ((t1 < 90) and (360-t2 < 90)):
#                 ## puts points at discontinuity from 0 to 360
#                 t2cont = t2 - 360
#                 p1 = (tnau[jt],tnau0[jt])
#                 p2 = (tnau[jt+1],t2cont)
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (0 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,0))

#                 t1cont = t1 + 360
#                 p1 = (tnau[jt],t1cont)
#                 p2 = (tnau[jt+1],tnau0[jt+1])
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 XX = (360 - lr.intercept) / lr.slope
#                 tnau_new = np.hstack((tnau_new,XX))
#                 tnau0_new = np.hstack((tnau0_new,360))

#             else:
#                 if t1 < t2:
#                     kk = np.where((degbins > t1) & (degbins <t2))[0]
#                 else:
#                     kk = np.where((degbins > t2) & (degbins <t1))[0]
#                 if np.any(kk):
#                     p1 = (tnau[jt],tnau0[jt])
#                     p2 = (tnau[jt+1],tnau0[jt+1])
#                     xx = [p1[0],p2[0]]
#                     yy = [p1[1],p2[1]]
#                     lr = ss.linregress(xx,yy)
#                     dbkk = degbins[kk]
#                     XX = (dbkk - lr.intercept) / lr.slope
#                     tnau_new = np.hstack((tnau_new,XX))
#                     tnau0_new = np.hstack((tnau0_new,dbkk))
#     nt = len(tnau_new)

#     tnau_new2 = np.array([])
#     tnau0_new2 = np.array([])
#     # SS_new = np.array([])
#     for jt in range(nt-1):
#         t1 = tnau0_new[jt]
#         t2 = tnau0_new[jt+1]
#         tnau_new2 = np.hstack((tnau_new2,tnau_new[jt]))
#         tnau0_new2 = np.hstack((tnau0_new2,tnau0_new[jt]))

#         if not np.abs(t1-t2) == 360:
#             if t1 < t2:
#                 kk = np.where((degbins > t1) & (degbins <t2))[0]
#             else:
#                 kk = np.where((degbins > t2) & (degbins <t1))[0]
#             if np.any(kk):
#                 p1 = (tnau_new[jt],tnau0_new[jt])
#                 p2 = (tnau_new[jt+1],tnau0_new[jt+1])
#                 xx = [p1[0],p2[0]]
#                 yy = [p1[1],p2[1]]
#                 lr = ss.linregress(xx,yy)
#                 dbkk = degbins[kk]
#                 XX = (dbkk - lr.intercept) / lr.slope
#                 tnau_new2 = np.hstack((tnau_new2,XX))
#                 tnau0_new2 = np.hstack((tnau0_new2,dbkk))

#     if tnau[0] > tnau[-1]:
#         tnau = np.flipud(tnau)
#         tnau_new2 = np.flipud(tnau_new2)
#         tnau0_new2 = np.flipud(tnau0_new2)
#         SS = np.flipud(SS)
#     SS_new = np.interp(tnau_new2,tnau,SS)
#     # nt = len(tnau_new)
#     # for jt in range(nt-1):
#     #     p1ss = (tnau_new[jt],SS[jt])
#     #     p2ss = (tnau[jt+1],SS[jt+1])
#     #     xxss = [p1ss[0],p2ss[0]]
#     #     yyss = [p1ss[1],p2ss[1]]
#     #     lrss = ss.linregress(xxss,yyss)
#     #     XXss = (dbkk - lrss.intercept) / lrss.slope
#     # SS_new = np.hstack((SS_new,XXss))

#     return tnau_new2, tnau0_new2, SS_new


# def compute_transformation(degbins,tnau,tnau0,sca):
#     """
#     Compute transformation ratio of E / Eo.
#     Input:
#     -----
#     degbins = directional bin boundaries
#     tnau = sheltered theta [degrees]
#     tnau0 = offshore theta [degrees]
#     sca   = scaling factor (k/ko)*(Cgo/Cg)

#     Return:
#     ------
#     RS = Transformation for each directional bin
#     PM = Ratio of energy that traveled through a caustic to total energy

#     """

#     rar = np.zeros(len(degbins)-1)
#     rarM = np.zeros(len(degbins)-1)
#     rarP = np.zeros(len(degbins)-1)
#     RS = np.zeros(len(degbins)-1)
#     SS_mn = np.zeros(len(degbins)-1)
#     PM = np.zeros(len(degbins)-1)

#     for jd in range(len(degbins)-1):
#         d1 = degbins[jd]
#         d2 = degbins[jd+1]

#         kk = np.where((tnau0>=d1) & (tnau0 <= d2))[0]
#         if np.any(kk):

#             #### identifies clusters of points in kk range ####
#             count = 0
#             for jb in range(len(kk)):
#                 kki = kk[jb]
#                 if jb == 0:
#                     inds = []
#                     inds.append([])
#                     inds[count].append(kki)
#                 else:
#                     if kki == (inds[-1][-1] + 1):

#                         inds[count].append(kki)
#                     else:
#                         count = count + 1
#                         inds.append([])
#                         inds[count].append(kki)
#             #####################################
#             #### identifies ranges of indicies ####
#             ratio = np.ones(len(inds))*np.nan
#             ratioM = np.ones(len(inds))*np.nan
#             ratioP = np.ones(len(inds))*np.nan
#             SS_seg = np.ones(len(inds))*np.nan

#             for ji in range(len(inds)):
#                 imin = np.min(inds[ji])
#                 imax = np.max(inds[ji])+1
#                 tnaut = tnau[imin:imax]
#                 tnau0t = tnau0[imin:imax]
#                 scat = sca[imin:imax]
#                 if ((tnau0[imin] == d1) and (tnau0[imax-1] == d1)) or ((tnau0[imin] == d2) and (tnau0[imax-1] == d2)):
#                     ## if caustic occurs (within theta bin)

#                     dth0 = np.diff(tnau0t)
#                     dth = np.diff(tnaut)
#                     kkp = (dth0 > 0)
#                     kkm = (dth0 <= 0)

#                     dth0p = dth0[kkp]
#                     dth0m = dth0[kkm]

#                     dthp = dth[kkp]
#                     dthm = dth[kkm]

#                     dth = np.diff(tnaut)
#                     dth0 =  np.max(tnau0t) - np.min(tnau0t)
#                     ratio[ji] = np.nansum(np.abs(dth))/ dth0
#                     if any(kkp):
#                         ratioP[ji] = np.nansum(np.abs(dthp)) / dth0
#                     if any(kkm):
#                         ratioM[ji] = np.nansum(np.abs(dthm)) / dth0


#                 else:

#                     dth0 = np.diff(tnau0t)
#                     dth = np.diff(tnaut)
#                     dth0 =  np.max(tnau0t) - np.min(tnau0t)

#                     ratio[ji] = np.nansum(np.abs(dth)) / dth0#np.nansum(np.abs(dth0))
#                     SS_seg[ji] = np.nanmean(scat)

#                     kkm = (dth0 <= 0)
#                     kkp = (dth0 > 0)

#                     if np.any(kkm):
#                         ratioM[ji] = np.nansum(np.abs(dth[kkm])) / dth0 #np.nansum(np.abs(dth0[kkm]))
#                     if np.any(kkp):
#                         ratioP[ji] = np.nansum(np.abs(dth[kkp])) / dth0 #np.nansum(np.abs(dth0[kkp]))

#                     # print(ratio[ji],np.nansum([ratioM[ji],ratioP[ji]]))

#             SS_mn[jd] = np.nanmean(SS_seg)
#             RS[jd] = np.nansum(ratio*SS_seg)
#             rar[jd] = np.nansum(ratio)
#             rarM[jd] = np.nansum(ratioM)
#             rarP[jd] = np.nansum(ratioP)

#             PM[jd] = np.nansum(ratioM) / (np.nansum(ratioM) + np.nansum(ratioP))
#     return RS,PM


def make_empty_matrices(freq,deg):
    """
    Makes empty matrices to be filled
    Input:
    freq = frequencies that will be in matrix
    deg = directions that will be in matrix

    Return:
    ------
    TM = matrix for transformation matrix
    CM = matrix for caustic matrix
    SS = matrix for (k/ko)*(Cgo/Cg)
    KRA = matrix for (k/ko)
    REFS = matrix for (k/ko)*(dtheta/dthetao)
    CGRA = matrix for (Cgo/Cg)
    rar = matrix for (dtheta/dthetao)
    """
    nfreq = len(freq)
    ndeg = len(deg)
    TM = np.ones((nfreq,ndeg))*np.nan
    CM = np.zeros((nfreq,ndeg))
    SS = np.ones((nfreq,ndeg))*np.nan
    KRA = np.ones((nfreq,ndeg))*np.nan
    REFS = np.ones((nfreq,ndeg))*np.nan
    CGRA = np.ones((nfreq,ndeg))*np.nan
    rar = np.ones((nfreq,ndeg))*np.nan

    return TM,CM,SS,KRA,REFS,CGRA,rar

def write_transformation_matrix(freq,deg,degbins):
    """
    Make transformation matrix (E/Eo).
    Input:
    ------
    freq    = wave frequency
    deg     = wave directions, upper bound (degrees, nautical direction)
    degbins = wave directional bin bounds (deg, but includes first bin)

    Return:
    ------
    TM    = Transformation matrix [E / Eo] [nf x nd]
    CM    = Caustic Metric (Negative to total energy ratio) [nf x nd]

    """
    TM = np.ones((nfreq,ndeg))*np.nan
    CM = np.zeros((nfreq,ndeg))
    for jf in range(len(freq)):
        nfreq = len(freq)
        ndeg = len(deg)
        tp = 1/freq[jf]
        XOUT,XXOUT,TTHETA,TTHETAO,FLAG,XOUT_R1,XOUT_R2,grid,Z,SS=palau_quick_test(tp,output=True)

        ###################################################################
        # Convert from radians & math to degrees and nautical direction
        tnau,tnau0 = get_theta(TTHETA[:-1],TTHETAO[:-1])
        sca = SS[:-1]
        fla = FLAG[:-1]
        ###################################################################
        ###################################################################
        # inserts NaN where rays strike land
        tnau,tnau0,sca,fla = fill_islands(tnau,tnau0,sca,fla)
        ###################################################################


        ###################################################################
        # linear interpolation to make sure bin widths are included
        tnau,tnau0,sca = LI2degbin(degbins,tnau,tnau0,sca)
        ###################################################################

        ###################################################################
        # compute transformation functions
        RS,PM = compute_transformation(degbins,tnau,tnau0,sca)
        ###################################################################
        # fill matrix
        TM[jf,:] = RS
        CM[jf,:] = PM
        print("TP = %.02f"%tp)

    return TM, CM


def plot_dirSpec(dirSpec, freq, directions,title=[], vmin=0,vmax=.04,saveas=False,figpath=[],svname=None):
    """Plots the directional spectrum

        Input:
            dirSpec = directional spectrum with shape [directions, frequencies]
            freq = frequencies
    """
    Ndir = dirSpec.shape[0]
    limits = np.linspace(vmin,vmax,30)

    # if directions == None:
    #     azimuths = np.radians(np.linspace(0, 360, Ndir))
    # else:
    #     azimuths = directions
    azimuths = directions
    ff,dd = np.meshgrid(freq, azimuths)
    extend = "max"

    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(projection='polar'))
    cmap = cm.get_cmap("jet").copy()
    cmap.set_under(color='white')
    cs = ax.contourf(dd,ff,dirSpec,levels=limits,cmap=cmap,extend=extend)
    ax.set_rmax(.28)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    thetaticks = np.arange(0,360,30)
    thetalabels = [str(s)+'$^o$' for s in np.arange(0,360,30)]
    thetalabels[0] = '360'+'$^o$'
    ax.set_thetagrids(thetaticks, thetalabels)
    periods = np.array([20,12,8,6,4])
    rticks = 1./periods
    rlabels = [str(p)+' s' for p in periods]
    ax.set_rgrids(rticks)
    ax.set_rlabel_position(130)
    cbar = plt.colorbar(cs, orientation='horizontal',fraction=0.04, format='%0.2f',ticks=np.arange(0,3,.5))
    cbar.ax.tick_params(labelsize=20)
    ax.set_yticklabels(rlabels, fontsize=12,color="w")
    ax.tick_params(labelsize=20)
    cbar.set_label(' $\\frac{E(f)}{E_{o}(f,\\theta_{o})}$',fontsize=20, labelpad =14)
    ax.set_title(title,fontsize=20)
    if saveas:
        print('saving figure on %s' %svname)
        fig.savefig(figpath + svname, dpi=300)

    return

def make_metadata_dict(latitude,longitude,depth,shoreNormal,bathyOffset,shoreFlag,label):
    """
    Make a dictionary of meta data required to run Fortran Spectral Refraction Model.

    Input:
    -----
    latitude = latitude of point
    longitude = longitude of point
    depth  = depth of point
    shoreNormal = direction of shore normal
    bathyOffset = Offset to bathymetry grid
    shoreFlag = [0 1 2 3]
        0 = coastal structures
        1 = planar bathymetry
        2 = complex bathymetry
        3 = offshore or unspecified
    label = site label

    Output:
    ------
    meta = dictionary of metadata variables
    """
    meta = dict()
    meta["latitude"] = latitude
    meta["longitude"] = longitude
    meta["depth"] = depth
    meta["shoreNormal"] = shoreNormal
    meta["bathyOffset"] = bathyOffset
    meta["shoreFlag"] = shoreFlag
    meta["label"] = label

    return meta

def compute_moments(deg,TM):
    """
    Computes directional moments from transformation matrix.
    Input:
    -----
    deg = wave direction (nautical direction from 1-360)
    TM  = transformation matrix

    Output:
    ------
    seav = sea directional moments
    swlv = swell directional moments
    """

    theta = np.deg2rad(deg)

    m1 = 1
    m2 = np.cos(theta)
    m3 = np.sin(theta)
    m4 = np.cos(2*theta)
    m5 = np.sin(2*theta)

    swl_et = TM*m1
    sea_et = TM*m1

    swl_ec = TM*m2
    swl_es = TM*m3
    swl_ec2 = TM*m4
    swl_es2 = TM*m5

    sea_ec = TM*m2
    sea_es = TM*m3
    sea_ec2 = TM*m4
    sea_es2 = TM*m5

    seav = dict(sea_et=sea_et,sea_ec=sea_ec,sea_es=sea_es,sea_ec2=sea_ec2,sea_es2=sea_es2)
    swlv = dict(swl_et=swl_et,swl_ec=swl_ec,swl_es=swl_es,swl_ec2=swl_ec2,swl_es2=swl_es2)
    return seav,swlv

def save_tmatrix(svpath,svname,svpath_mn,svname_mn,senname,freq,deg,TM,CM,meta):
    write_nc_tmatrix(svpath,svname,senname,freq,deg,TM,CM,meta)
    print("netCDF at 1 degree resolution saved!")
    degmn,TMmn = mean_dir_bins(TM)
    degmn,CMmn = mean_dir_bins(CM)
    write_nc_tmatrix(svpath_mn,svname_mn,senname,freq,degmn,TMmn,CMmn,meta)
    print("netCDF at 5 degree resolution saved!")


def write_nc_tmatrix(svpath,svname,senname,freq,deg,TM,CM,meta):

    seav,swlv = compute_moments(deg,TM)

    nfreq = len(freq)
    ndeg = len(deg)
    ncf = nc.Dataset(svpath + svname,mode='w',format='NETCDF4')

    # create dimensions
    seafreq_dim = ncf.createDimension('seaFrequency', nfreq)
    swellfreq_dim = ncf.createDimension('swellFrequency', nfreq)
    strlen_dim = ncf.createDimension('maxStrlen64', 5)

    deg_dim = ncf.createDimension('direction', ndeg)
    ncf.site_name='%s' % senname

    # create coordinates
    seafreqnc = ncf.createVariable('seaFrequency', np.float32, ('seaFrequency',))
    seafreqnc.units = "Hertz"
    seafreqnc.long_name = "Sea Frequency"

    swlfreqnc = ncf.createVariable('swellFrequency', np.float32, ('swellFrequency',))
    swlfreqnc.units = "Hertz"
    swlfreqnc.long_name = "Swell Frequency"

    degnc = ncf.createVariable('direction', np.float32, ('direction',))
    degnc.units = "degrees"
    degnc.long_name = "Wave Direction"

    # create meta data variables
    latnc = ncf.createVariable('latitude', np.float32, )
    latnc.units = "degrees"
    latnc.long_name = "latitude"

    lonnc = ncf.createVariable('longitude', np.float32, )
    lonnc.units = "degrees"
    lonnc.long_name = "longitude"

    depnc = ncf.createVariable('depth', np.float32, )
    depnc.units = "meters"
    depnc.long_name = "water depth"

    BOnc = ncf.createVariable('bathyOffset', np.float32, )
    BOnc.units = "meters"
    BOnc.long_name = "Offset to bathymetry grid"
    BOnc.comment = "Offset added to the input bathymetry grid, by default the MLLW-to-MSL offset for the site."

    SFnc = ncf.createVariable('shoreFlag', np.int8, )
    SFnc.long_name = "shoreline classification flag"
    SFnc.flag_values = np.array([0,1,2,3],dtype=np.int8)
    SFnc.flag_meanings = 'coastal_structures planar_bathymetry complex_bathymetry offshore_or_unspecified'

    SNnc = ncf.createVariable('shoreNormal', np.float32,fill_value=99999.99)
    SNnc.long_name = "shore normal"
    SNnc.units = "degrees"

    SLnc = ncf.createVariable('label', "S1", ('maxStrlen64',),fill_value=b"_")
    SLnc.long_name = "site_label"

    # create variables
    sea_et = ncf.createVariable('sea_et', np.float32, ('seaFrequency','direction'))
    sea_et.units = "sea m1"
    sea_et.long_name = "Transformation Matrix"

    sea_ec = ncf.createVariable('sea_ec', np.float32, ('seaFrequency','direction'))
    sea_ec.units = "sea m2"
    sea_ec.long_name = "Transformation Matrix"

    sea_es = ncf.createVariable('sea_es', np.float32, ('seaFrequency','direction'))
    sea_es.units = "sea m3"
    sea_es.long_name = "Transformation Matrix"

    sea_ec2 = ncf.createVariable('sea_ec2', np.float32, ('seaFrequency','direction'))
    sea_ec2.units = "sea m4"
    sea_ec2.long_name = "Transformation Matrix"

    sea_es2 = ncf.createVariable('sea_es2', np.float32, ('seaFrequency','direction'))
    sea_es2.units = "sea m5"
    sea_es2.long_name = "Transformation Matrix"

    swl_et = ncf.createVariable('swl_et', np.float32, ('swellFrequency','direction'))
    swl_et.units = "swl m1"
    swl_et.long_name = "Transformation Matrix"

    swl_ec = ncf.createVariable('swl_ec', np.float32, ('swellFrequency','direction'))
    swl_ec.units = "swl m2"
    swl_ec.long_name = "Transformation Matrix"

    swl_es = ncf.createVariable('swl_es', np.float32, ('swellFrequency','direction'))
    swl_es.units = "swl m3"
    swl_es.long_name = "Transformation Matrix"

    swl_ec2 = ncf.createVariable('swl_ec2', np.float32, ('swellFrequency','direction'))
    swl_ec2.units = "swl m4"
    swl_ec2.long_name = "Transformation Matrix"

    swl_es2 = ncf.createVariable('swl_es2', np.float32, ('swellFrequency','direction'))
    swl_es2.units = "swl m5"
    swl_es2.long_name = "Transformation Matrix"


    cmnc = ncf.createVariable('cm', np.float32, ('swellFrequency','direction'))
    cmnc.units = "Negative to Total Energy"
    cmnc.long_name = "Caustic Matrix"

    ## assign values for coordinates
    seafreqnc[:] = freq
    swlfreqnc[:] = freq
    degnc[:] = deg

    ## assign values for metadata
    latnc.assignValue(meta["latitude"])
    lonnc.assignValue(meta["longitude"])
    depnc.assignValue(meta["depth"])
    BOnc.assignValue(meta["bathyOffset"])
    SFnc.assignValue(meta["shoreFlag"])
    SNnc.assignValue(meta["shoreNormal"])
    # SLnc[:] = np.array(meta["label"].ljust(5), dtype='object')
    SLnc[:] = np.array(list(meta["label"]),dtype='S1')
    print(meta["label"])
    ## asign values for variables
    sea_et[:,:] = seav["sea_et"]
    sea_ec[:,:] = seav["sea_ec"]
    sea_es[:,:] = seav["sea_es"]
    sea_ec2[:,:] = seav["sea_ec2"]
    sea_es2[:,:] = seav["sea_es2"]

    swl_et[:,:] = swlv["swl_et"]
    swl_ec[:,:] = swlv["swl_ec"]
    swl_es[:,:] = swlv["swl_es"]
    swl_ec2[:,:] = swlv["swl_ec2"]
    swl_es2[:,:] = swlv["swl_es2"]

    cmnc[:,:] = CM
    ncf.close()

    print("File saved!")



def mean_dir_bins(EE):
    """
    Averages transformation matrix to be consistent with CDIP 5 degree directional bins
    Input:
    -----
    deg = direction
    EE  = transformation matrix

    Output:
    ------
    dirmn = mean directional


    """
    ndegbin = 72
    nf,ndeg = np.shape(EE)
    EE_mn = np.ones((nf,ndegbin))
    for jf in range(nf):
        for ii,jt in enumerate(range(2,355,5)):
            xx = EE[jf,:]
            inds = np.arange(jt,jt+5)
            EE_mn[jf,ii] = np.nanmean(xx[inds])
        ii1 = [0,1,357,358,359]
        EE_mn[jf,71] = np.nanmean(xx[ii1])

    dirmn = np.arange(5,365,5)

    return dirmn,EE_mn


def write_nc_physmatrix(svpath,svname,senname,freq,deg,TM,CM,SSMN,KRA,REFS,CGRA,rar,meta):
    nfreq = len(freq)
    ndeg = len(deg)
    ncf = nc.Dataset(svpath + svname,mode='w',format='NETCDF4')

    # create dimensions
    freq_dim = ncf.createDimension('frequency', nfreq)
    deg_dim = ncf.createDimension('direction', ndeg)
    strlen_dim = ncf.createDimension('maxStrlen64', 5)
    ncf.site_name='%s' % senname

    # create coordinates
    freqnc = ncf.createVariable('frequency', np.float32, ('frequency',))
    freqnc.units = "Hertz"
    freqnc.long_name = "Frequency"

    degnc = ncf.createVariable('direction', np.float32, ('direction',))
    degnc.units = "degrees"
    degnc.long_name = "Wave Direction"

    # create meta data variables
    latnc = ncf.createVariable('latitude', np.float32, )
    latnc.units = "degrees"
    latnc.long_name = "latitude"

    lonnc = ncf.createVariable('longitude', np.float32, )
    lonnc.units = "degrees"
    lonnc.long_name = "longitude"

    depnc = ncf.createVariable('depth', np.float32, )
    depnc.units = "meters"
    depnc.long_name = "water depth"

    BOnc = ncf.createVariable('bathyOffset', np.float32, )
    BOnc.units = "meters"
    BOnc.long_name = "Offset to bathymetry grid"
    BOnc.comment = "Offset added to the input bathymetry grid, by default the MLLW-to-MSL offset for the site."

    SFnc = ncf.createVariable('shoreFlag', np.int8, )
    SFnc.long_name = "shoreline classification flag"
    SFnc.flag_values = np.array([0,1,2,3],dtype=np.int8)
    SFnc.flag_meanings = 'coastal_structures planar_bathymetry complex_bathymetry offshore_or_unspecified'

    SNnc = ncf.createVariable('shoreNormal', np.float32,fill_value=99999.99)
    SNnc.long_name = "shore normal"
    SNnc.units = "degrees"

    SLnc = ncf.createVariable('label', "S1", ('maxStrlen64',),fill_value=b"_")
    SLnc.long_name = "site_label"

    # create variables
    tmnc = ncf.createVariable('TM', np.float32, ('frequency','direction'))
    tmnc.units = "--"
    tmnc.long_name = "Transformation Matrix (E/Eo)"

    cmnc = ncf.createVariable('CM', np.float32, ('frequency','direction'))
    cmnc.units = "Negative to Total Energy"
    cmnc.long_name = "Caustic Matrix"

    SS = ncf.createVariable('SS', np.float32, ('frequency','direction'))
    SS.units = "--"
    SS.long_name = "SS (k/ko) * (Cgo /Cg)"

    kra = ncf.createVariable('Kra', np.float32, ('frequency','direction'))
    kra.units = "--"
    kra.long_name = "Refraction Coefficient k/ko"

    refs = ncf.createVariable('Refs', np.float32, ('frequency','direction'))
    refs.units = "--"
    refs.long_name = "Theta Ref Coefficient (dtheta/dthetao)*k/ko"

    cgra = ncf.createVariable('Cgra', np.float32, ('frequency','direction'))
    cgra.units = "--"
    cgra.long_name = "Shoaling Coefficient Cgo/Cg"

    dtra = ncf.createVariable('DTra', np.float32, ('frequency','direction'))
    dtra.units = "--"
    dtra.long_name = "Dtheta / Dthetao"


    ## assign values for coordinates
    freqnc[:] = freq
    degnc[:] = deg

    ## assign values for metadata
    latnc.assignValue(meta["latitude"])
    lonnc.assignValue(meta["longitude"])
    depnc.assignValue(meta["depth"])
    BOnc.assignValue(meta["bathyOffset"])
    SFnc.assignValue(meta["shoreFlag"])
    SNnc.assignValue(meta["shoreNormal"])
    # SLnc[:] = np.array(meta["label"].ljust(5), dtype='object')
    SLnc[:] = np.array(list(meta["label"]),dtype='S1')
    print(meta["label"])
    ## asign values for variables
    tmnc[:,:] = TM
    cmnc[:,:] = CM
    SS[:,:] = SSMN
    kra[:,:] = KRA
    refs[:,:] = REFS
    cgra[:,:] = CGRA
    dtra[:,:] = rar



    ncf.close()

    print("File saved!")



def compute_transformation(degbins,tnau,tnau0,sca,Kratio=[],Cgratio=[]):
    """
    Compute transformation ratio of E / Eo.
    Input:
    -----
    degbins = directional bin boundaries
    tnau = sheltered theta [degrees]
    tnau0 = offshore theta [degrees]
    sca   = scaling factor (k/ko)*(Cgo/Cg)

    Return:
    ------
    RS = Transformation for each directional bin
    PM = Ratio of energy that traveled through a caustic to total energy

    """

    rar = np.zeros(len(degbins)-1)
    rarM = np.zeros(len(degbins)-1)
    rarP = np.zeros(len(degbins)-1)
    RS = np.zeros(len(degbins)-1)
    SS_mn = np.zeros(len(degbins)-1)
    PM = np.zeros(len(degbins)-1)
    REFS = np.zeros(len(degbins)-1)
    if np.any(Kratio):
        Kratio_mn = np.zeros(len(degbins)-1)
    if np.any(Cgratio):
        Cgratio_mn = np.zeros(len(degbins)-1)

    for jd in range(len(degbins)-1):
        d1 = degbins[jd]
        d2 = degbins[jd+1]

        kk = np.where((tnau0>=d1) & (tnau0 <= d2))[0]
        if np.any(kk):

            #### identifies clusters of points in kk range ####
            count = 0
            for jb in range(len(kk)):
                kki = kk[jb]
                if jb == 0:
                    inds = []
                    inds.append([])
                    inds[count].append(kki)
                else:
                    if kki == (inds[-1][-1] + 1):

                        inds[count].append(kki)
                    else:
                        count = count + 1
                        inds.append([])
                        inds[count].append(kki)
            #####################################
            #### identifies ranges of indicies ####
            ratio = np.ones(len(inds))*np.nan
            ratioM = np.ones(len(inds))*np.nan
            ratioP = np.ones(len(inds))*np.nan
            SS_seg = np.ones(len(inds))*np.nan

            if np.any(Kratio):
                Kratio_seg = np.ones(len(inds))*np.nan
            if np.any(Cgratio):
                Cgratio_seg = np.ones(len(inds))*np.nan
            for ji in range(len(inds)):
                imin = np.min(inds[ji])
                imax = np.max(inds[ji])+1
                tnaut = tnau[imin:imax]
                tnau0t = tnau0[imin:imax]
                scat = sca[imin:imax]
                if np.any(Kratio):
                    krt = Kratio[imin:imax]
                if np.any(Cgratio):
                    cgrt = Cgratio[imin:imax]
                if ((tnau0[imin] == d1) and (tnau0[imax-1] == d1)) or ((tnau0[imin] == d2) and (tnau0[imax-1] == d2)):
                    ## if caustic occurs (within theta bin)

                    dth0 = np.diff(tnau0t)
                    dth = np.diff(tnaut)
                    kkp = (dth0 > 0)
                    kkm = (dth0 <= 0)

                    dth0p = dth0[kkp]
                    dth0m = dth0[kkm]

                    dthp = dth[kkp]
                    dthm = dth[kkm]

                    dth = np.diff(tnaut)
                    dth0 =  np.max(tnau0t) - np.min(tnau0t)
                    ratio[ji] = np.nansum(np.abs(dth))/ dth0
                    SS_seg[ji] = np.nanmean(scat)
                    if np.any(Kratio):
                        Kratio_seg[ji] = np.nanmean(krt)
                    if np.any(Cgratio):
                        Cgratio_seg[ji]= np.nanmean(cgrt)
                    if any(kkp):
                        ratioP[ji] = np.nansum(np.abs(dthp)) / dth0
                    if any(kkm):
                        ratioM[ji] = np.nansum(np.abs(dthm)) / dth0

                else:

                    dth0 = np.diff(tnau0t)
                    dth = np.diff(tnaut)
                    dth0 =  np.max(tnau0t) - np.min(tnau0t)

                    ratio[ji] = np.nansum(np.abs(dth)) / dth0#np.nansum(np.abs(dth0))
                    SS_seg[ji] = np.nanmean(scat)
                    if np.any(Kratio):
                        Kratio_seg[ji] = np.nanmean(krt)
                    if np.any(Cgratio):
                        Cgratio_seg[ji]= np.nanmean(cgrt)
                    kkm = (dth0 <= 0)
                    kkp = (dth0 > 0)

                    if np.any(kkm):
                        ratioM[ji] = np.nansum(np.abs(dth[kkm])) / dth0 #np.nansum(np.abs(dth0[kkm]))
                    if np.any(kkp):
                        ratioP[ji] = np.nansum(np.abs(dth[kkp])) / dth0 #np.nansum(np.abs(dth0[kkp]))

                    # print(ratio[ji],np.nansum([ratioM[ji],ratioP[ji]]))

            SS_mn[jd] = np.nanmean(SS_seg)
            if np.any(Kratio):
                Kratio_mn[jd] = np.nanmean(Kratio_seg)
                REFS[jd] = np.nansum(ratio*Kratio_seg)
            if np.any(Cgratio):
                Cgratio_mn[jd] = np.nanmean(Cgratio_seg)
            RS[jd] = np.nansum(ratio*SS_seg)
            rar[jd] = np.nansum(ratio)
            rarM[jd] = np.nansum(ratioM)
            rarP[jd] = np.nansum(ratioP)
            PM[jd] = np.nansum(ratioM) / (np.nansum(ratioM) + np.nansum(ratioP))
    if np.any(Kratio) and not np.any(Cgratio):
        return RS,PM,SS_mn,Kratio_mn,REFS
    elif np.any(Kratio) and np.any(Cgratio):
        return RS,PM,SS_mn,Kratio_mn,REFS,Cgratio_mn,rar
    elif np.any(Cgratio) and not np.any(Kratio):
        return RS,PM,SS_mn,Cgratio_mn
    elif not np.any(Cgratio) and not np.any(Kratio):
        return RS,PM
