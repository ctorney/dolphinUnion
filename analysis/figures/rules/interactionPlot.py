from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator, FixedLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot
from matplotlib import transforms
from mpl_toolkits.axisartist import Subplot
import mpl_toolkits.axisartist.floating_axes as floating_axes


import os
import csv
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import pandas as pd
import math
from math import *

#from viridis import viridis
#plt.register_cmap(name='viridis', cmap=viridis)
#viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
#plt.register_cmap(name='viridis_r', cmap=viridis_r)


 
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
    fig.add_subplot(ax1)
 
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
 
    ax1.axis["left"].label.set_text("distance (meters)")
    ax1.axis["top"].label.set_text("")
 
    # create a parasite axes
    aux_ax = ax1.get_aux_axes(tr)
 
    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                         # drawn twice, and possibly over some other
                         # artists. So, we decrease the zorder a bit to
                         # prevent this.
 
    return ax1, aux_ax

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

# 
## now let's plot the finished product!
fig1=plt.figure(figsize=(8,8))
#fig = plt.figure(1, figsize=(8, 4))
fig1.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
 
ax1, aux_ax1 = setup_axes(fig1, 111, theta=[thetamin,thetamax], radius=[rmin, rmax])
 
#ax2 = fractional_polar_axes(fig1 ,thlim=(-degrees(ia), degrees(ia)),rlim=(sr,maxr),step=(degrees(2*ia/(binn1+1)),dr))
#ax2 = fractional_polar_axes(fig1 ,thlim=(-degrees(ia), degrees(ia)),rlim=(sr,maxr),step=(degrees(2*ia/(binn1+1)),dr))
# example spiral plot:
#thstep = 10
#th = np.arange(0, 180+thstep, thstep) # deg
#rstep = 1/(len(th)-1)
#r = np.arange(0, 1+rstep, rstep)
#a1.plot(th, r, 'b')
#f1.show()

#ax2=plt.subplot(projection="polar",frameon=False)
ax1.grid()
im=aux_ax1.pcolormesh(X,Y,img,lw=1.0,vmin=0,vmax=1.0,cmap='viridis')
#im=ax2.pcolormesh(hista2,vmin=0,vmax=np.max(hista2),cmap='viridis')
#im=ax2.pcolormesh(2*y,x,z,vmin=0,vmax=np.max(hista2),cmap='viridis')
#ax2.xaxis.set_visible(False)

## angle lines
#ax2.set_thetagrids(angles=np.arange(0),labels=['','10', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
#
#ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
##ax1.yaxis.set_visible(False)
##ax1.set_ylim(0,0.5)
#ax2.yaxis.set_visible(False)
##ax1.set_thetagrids(angles=np.arange(0,36,4.5),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
#ax1.set_thetagrids(angles=np.degrees(np.arange(-0.4,0.5,0.2)),labels=['-0.4','-0.2', '0', '0.2', '0.4'],frac=1.1)
##ax1.set_rgrids(radii=np.arange(1,6,1),labels=['-0.4','-0.2', '0', '0.2', '0.4'],frac=1.1)
#ax1.set_rgrids(radii=np.arange(1,6,0.9),labels=['1','2','3','4','5',''], angle=22.918)
##ax1.set_rgrids(radii=np.arange(1,6,0.1))
#colourbar
position=fig1.add_axes([1.1,0.22,0.02,0.5])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Social weighting', rotation=90,fontsize='large',labelpad=15)      
#
##body length legend - draws the ticks and 
#axes=ax2            
#factor = 0.98
#d = axes.get_yticks()[-1] #* factor
#r_tick_labels = [0] + axes.get_yticks()
#r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
#r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#   
## fixed offsets in x
#offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
#offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
#offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
#trans_spine = axes.transData + offset_spine
#trans_ticklabels = trans_spine + offset_ticklabels
#trans_axlabel = trans_spine + offset_axlabel
#axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
#
## plot the 'tick labels'
#for ii in range(len(r_ticks)):
#    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
#axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
#
#

fig1.savefig("rules.png",bbox_inches='tight',dpi=300)
fig1.savefig("rules.tiff",bbox_inches='tight',dpi=300)
# plot the 'spine'

    