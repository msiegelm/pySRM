import matplotlib.pyplot as plt
import numpy as np

def plot_rays(XXOUT,TTHETA,TTHETAO,FLAG,SS,XOUT_R1,XOUT_R2,grd,saveas=False,figpath=[],svname=None):
    fig,ax=plt.subplots(figsize=(5,8))   
    pc=ax.pcolormesh(grd["x"],grd["y"],grd["z"].T, cmap='gist_earth_r', vmin=0, vmax=4000)
    fig.colorbar(pc)
    ax.contour(grd["x"],grd["y"],grd["z"].T, levels=[0,5,10,25,50,100, 1000, 4000], colors='k')
    n = len(XXOUT)
    colors = plt.cm.jet(np.linspace(0,1,n))
    ax.set_aspect("equal")
    for i in range(len(XXOUT)):
        ax.plot(XXOUT[i][:, 0], XXOUT[i][:, 1], color=colors[i])
    ax.plot(XOUT_R1[:, 0], XOUT_R1[:, 1], 'g')
    ax.plot(XOUT_R2[:, 0], XOUT_R2[:, 1], 'r')
    # for i in kk:
    #     ax.plot(XXOUT[i][:, 0], XXOUT[i][:, 1], color=colors[i])
    # ax.set_ylim([-50000,50000])
    if saveas:
        print('saving figure on %s' %svname)
        fig.savefig(figpath + svname, dpi=300)



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




def plot_tmatrix():
    plt.rcParams["font.size"] = 16
    fig,ax=plt.subplots(figsize=(8,5))
    pc1=ax.pcolormesh(deg,freq,CM*100,vmin=0,vmax=100,cmap="jet")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Wave Direction [$^\\circ$]")
    cb=fig.colorbar(pc1,ax=ax)
    cb.set_label("Negative : Total Energy")
    ax.set_xlim([0,360])
    ax.set_title("Caustic Metric")
    fig.savefig(figpath + "pl_causticmetric_%s.png"%senname, dpi=300, bbox_inches='tight',)

    fig,ax=plt.subplots(figsize=(8,5))
    pc=ax.pcolormesh(deg,freq,TM,vmin=0,vmax=2,cmap="jet")
    ax.set_title("New")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Wave Direction [$^\\circ$]")
    cb=fig.colorbar(pc,ax=ax)
    cb.set_label("$\\frac{E}{E_o}$")
    ax.set_xlim([0,360])

    fig.savefig(figpath + "pl_tmatrix_compare_%s.png"%senname, dpi=300, bbox_inches='tight',)

