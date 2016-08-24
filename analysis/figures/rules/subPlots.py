import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
from math import pi
from math import *
import scipy.stats
from scipy.stats import binned_statistic_2d
from scipy.ndimage.filters import gaussian_filter1d
import constantModelAlign
import decayModelAlign
import networkModelAlign
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import transforms
#from viridis import viridis
from scipy.stats import binned_statistic_2d
from scipy.stats import norm

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator, FixedLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot
from matplotlib import transforms
from mpl_toolkits.axisartist import Subplot
import mpl_toolkits.axisartist.floating_axes as floating_axes

###############################################################################
###############################################################################
###############################################################################
###############################################################################
def setup_axes(fig, rect, theta, radius):
 
    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D() + PolarAxes.PolarTransform()
 
    # Find grid values appropriate for the coordinate (degree).
    # The argument is an approximate number of grids.
    grid_locator1 = angle_helper.LocatorD(5)
    grid_locator1 = FixedLocator([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
 
    # And also use an appropriate formatter:
    tick_formatter1 = angle_helper.FormatterDMS()
 
    # set up number of ticks for the r-axis
    grid_locator2 = MaxNLocator(8)
 
    # the extremes are passed to the function
    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                extremes=(theta[0], theta[1], radius[0], radius[1]),
                                grid_locator1=grid_locator1,
                                grid_locator2=grid_locator2,
                                tick_formatter1=None,
                                tick_formatter2=None,
                                )
  #  ax1 = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)#,figsize=[2,2])
 
    # adjust axis
    # the axis artist lets you call axis with
    # "bottom", "top", "left", "right"
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")
 
    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
 
    ax1.axis["left"].label.set_text("distance (meters)")#,fontsize='x-large')
    ax1.axis["left"].label.set_fontsize('x-large')
    ax1.axis["top"].label.set_text("")
 
    # create a parasite axes
    aux_ax = ax1.get_aux_axes(tr)
 
    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                         # drawn twice, and possibly over some other
                         # artists. So, we decrease the zorder a bit to
                         # prevent this.
 
    return ax1, aux_ax
    
###############################################################################
###############################################################################    
###############################################################################    
#plt.register_cmap(name='viridis', cmap=viridis)
#viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
#plt.register_cmap(name='viridis_r', cmap=viridis_r)
plt.close('all')


fig = plt.figure(figsize=(18, 12))
#pax1 = plt.subplot2grid((2,2), (0,0), projection="polar",frameon=False)
pax2 = plt.subplot2grid((2,2), (0,1))
pax3 = plt.subplot2grid((2,2), (1, 0))
pax4 = plt.subplot2grid((2,2), (1, 1))


#plt.tight_layout()#pad=2.5, w_pad=3.0, h_pad=2.50)


###############################################################################
###############################################################################    
###############################################################################  
###############################################################################
############################ PART A ###########################################    
###############################################################################
###############################################################################
###############################################################################    
###############################################################################  



ie = 9.273
il = 4.994
ia = 0.276
ig = 1.307
    
# input file grid dimensions
xmax = 500
ymax = 500
thetamin = -0.35
thetamax = 0.35
rmin = 0.0
rmax = 8.0
 

img = np.zeros( (xmax,ymax) )
for x in range(0,xmax):
    for y in range(0,ymax):
        
        rr = x*rmax/float(xmax)
        th = thetamin + (y/float(ymax))*(thetamax-thetamin)
        if rr>ig and th >-ia and th < ia:
            img[x,y] = np.exp(-((rr/il)**ie)) #np.sin(0.05*x) * np.sin(0.05*y) + 10
        
 
theta,rad = np.meshgrid(np.linspace(thetamin,thetamax,ymax),
np.linspace(rmin,rmax,xmax))
X = theta
Y = rad

 

 
ax1, aux_ax1 = setup_axes(fig, 221, theta=[thetamin,thetamax], radius=[rmin, rmax])

ax1.grid()
im=aux_ax1.pcolormesh(X,Y,img,lw=1.0,vmin=0,vmax=1.0,cmap='viridis')

pos1 = ax1.get_position() # get the original position 
pos2 = [pos1.x0 + 0.07 , pos1.y0 + 0.04 ,  pos1.width * 0.8, pos1.height * 0.8] 
ax1.set_position(pos2)
position=fig.add_axes([aux_ax1._position.bounds[0]-0.04,aux_ax1._position.bounds[1]-0.02,0.015,0.32])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Social weighting', rotation=90,fontsize='x-large',labelpad=15)      
#ax1.text(-0.2, 1.1, 'A', transform=ax1.transAxes, fontsize=24, va='top', ha='right')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.yaxis.set_ticks_position('left')
###############################################################################
###############################################################################    
###############################################################################  
###############################################################################
############################ PART B ###########################################    
###############################################################################
###############################################################################
###############################################################################    
###############################################################################  


iAlpha = np.load('../../movement/variation/Ialphas.npy')
#iBeta = np.load('Ibetas.npy')
#iRho = np.load('Irhos.npy')
#iAngle = np.load('Iiw.npy')




#plt.figure()    

#440154FF #46327FFF #365C8DFF #277F8EFF #1FA187FF #4AC26DFF #9FDA3AFF #FDE725FF

pax2.hist(iAlpha[0],100,range=[0.0,0.6],label='adults',normed=True,histtype='stepfilled',color='midnightblue',alpha=0.9)
pax2.hist(iAlpha[2],100,range=[0.0,0.6],label='calves',normed=True,histtype='stepfilled',color='firebrick',alpha=0.9)
pax2.hist(iAlpha[1],100,range=[0.0,0.6],label='bulls',normed=True,histtype='stepfilled',color='#FDE725',alpha=0.9)

pax2.legend()

pax2.set_ylabel('posterior probability density',fontsize=16)
pax2.set_xlabel('weighting to social cues',fontsize=16)


###############################################################################
###############################################################################    
###############################################################################  
###############################################################################
############################ PART C & D########################################    
###############################################################################
###############################################################################
###############################################################################    
###############################################################################  


#cma = constantModelAlign.constantP()
#dma = decayModelAlign.decayP()
#nma = networkModelAlign.networkP()

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (214, 39, 40)]#
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)  

    
    
    
neighbours = np.load('../../movement/pdata/neighbours.npy')
mvector = np.load('../../movement/pdata/mvector.npy')
evector = np.load('../../movement/pdata/evector.npy')




nn1 = np.zeros_like(mvector)-1
nn2 = np.zeros_like(mvector)-1


for i,db in enumerate(neighbours):
    rowLoc = np.zeros((0,1)).astype(np.float32)
    for row in db:
        if row[0]>1.0 and row[0]<50.0 and row[1]>-0.27642330874196225 and row[1]<0.27642330874196225:
            rowLoc = np.vstack((rowLoc,row[0]))
    if len(rowLoc):
        nn1[i]=np.min(rowLoc)#-np.min(rowLoc)
    if len(rowLoc)>1:
        sndmin = np.partition(np.reshape(rowLoc,(len(rowLoc))),1)[1]
        nn2[i]=sndmin#-np.min(rowLoc)
        

# decay model versus constant model        
dectocon=dma-cma
# decay model versus network model
dectonet=dma-nma


                            

rmin=1.5
rmax=20.5
nbins=19
bs = (rmax-rmin)/nbins


### network model comparison

## nearest neighbour
# average
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
# standard error
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err/=np.sqrt(counts)
err/=2 # plot half the standard error
# smoothing
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

pax3.plot(newX, f2(newX), label='1st neighbor',color='midnightblue',linewidth=1.5)
pax3.fill_between(newX, uf2(newX),lf2(newX),facecolor='midnightblue',linewidth=0, alpha=0.25)


## 2nd nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])

err, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)
err/=2 # plot half the standard error

bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])


bmsmooth = gaussian_filter1d(bin_means,sigma=2.0*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')


pax3.plot(newX, f2(newX), label='2nd neighbor',color='firebrick',linewidth=1.5)
pax3.fill_between(newX, uf2(newX),lf2(newX),facecolor='firebrick',linewidth=0, alpha=0.1)


pax3.axhline(y=0, color='k')
pax3.set_xlabel('distance (meters)',fontsize=16)
pax3.set_ylabel('model difference',fontsize=16)
pax3.legend()
pax3.set_xlim([2,20])



## constant model comparison


## nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)
err/=2 # plot half the standard error


#smoothing
bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

pax4.plot(newX, f2(newX),color='midnightblue', label='1st neighbor',linewidth=1.5)
pax4.fill_between(newX, uf2(newX),lf2(newX),facecolor='midnightblue',linewidth=0, alpha=0.25)

## 2nd nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)
err/=2 # plot half the standard error

bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

pax4.plot(newX, f2(newX),color='firebrick', label='2nd neighbor',linewidth=1.5)
pax4.fill_between(newX, uf2(newX),lf2(newX),facecolor='firebrick',linewidth=0, alpha=0.1)


#plt.ylim([-0.08,0.08])
pax4.axhline(y=0, color='k')
pax4.set_xlabel('distance (meters)',fontsize=16)
pax4.set_ylabel('model difference',fontsize=16)
pax4.legend()
pax4.set_xlim([2,20])
ax1.text(-0.39, 1.23, 'A', transform=ax1.transAxes, fontsize=24, va='top', ha='right')
pax3.text(-0.1, 1.1, 'B', transform=pax2.transAxes, fontsize=24, va='top', ha='right')
pax3.text(-0.1, 1.1, 'C', transform=pax3.transAxes, fontsize=24, va='top', ha='right')
pax4.text(-0.1, 1.1, 'D', transform=pax4.transAxes, fontsize=24, va='top', ha='right')




plt.savefig("fig3.tiff",bbox_inches='tight',dpi=300)
plt.savefig("fig3.png",bbox_inches='tight',dpi=300)


#plt.plot(newX, f2(newX),color='midnightblue', label='1st neighbor',linewidth=1.5)
#plt.fill_between(newX, uf2(newX),lf2(newX),facecolor='midnightblue',linewidth=0, alpha=0.25)
#plt.plot(newX, f2(newX),color='goldenrod', label='2nd neighbor',linewidth=1.5)
#plt.fill_between(newX, uf2(newX),lf2(newX),facecolor='goldenrod',linewidth=0, alpha=0.1)

