from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot

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
from matplotlib import transforms
from viridis import viridis
plt.register_cmap(name='viridis', cmap=viridis)
viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
plt.register_cmap(name='viridis_r', cmap=viridis_r)


"""Demo of polar plot of arbitrary theta. This is a workaround for MPL's polar plot limitation
to a full 360 deg.

Based on http://matplotlib.org/mpl_toolkits/axes_grid/examples/demo_floating_axes.py
"""




def fractional_polar_axes(f, thlim=(0, 180), rlim=(0, 1), step=(30, 0.2),
                          thlabel='theta', rlabel='r', ticklabels=True):
    """Return polar axes that adhere to desired theta (in deg) and r limits. steps for theta
    and r are really just hints for the locators. Using negative values for rlim causes
    problems for GridHelperCurveLinear for some reason"""
    th0, th1 = thlim # deg
    r0, r1 = rlim
    thstep, rstep = step

    # scale degrees to radians:
    tr_scale = Affine2D().scale(np.pi/180., 1.)
    tr = tr_scale + PolarAxes.PolarTransform()
    theta_grid_locator = angle_helper.LocatorDMS((th1-th0) // thstep)
    r_grid_locator = MaxNLocator((r1-r0) // rstep)
    theta_tick_formatter = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr,
                                        extremes=(th0, th1, r0, r1),
                                        grid_locator1=theta_grid_locator,
                                        grid_locator2=r_grid_locator,
                                        tick_formatter1=theta_tick_formatter,
                                        tick_formatter2=None)

    a = FloatingSubplot(f, 111, grid_helper=grid_helper)
    f.add_subplot(a,projection='polar')

    # adjust x axis (theta):
    a.axis["bottom"].set_visible(False)
    a.axis["top"].set_visible(False)
    a.axis["left"].set_visible(False)
    a.axis["right"].set_visible(False)
    #a.axis["top"].set_axis_direction("bottom") # tick direction
    #a.axis["top"].toggle(ticklabels=ticklabels, label=bool(thlabel))
    #a.axis["top"].major_ticklabels.set_axis_direction("top")
    #a.axis["top"].label.set_axis_direction("top")

    # adjust y axis (r):
    #a.axis["left"].set_axis_direction("bottom") # tick direction
    #a.axis["right"].set_axis_direction("top") # tick direction
    #a.axis["left"].toggle(ticklabels=ticklabels, label=bool(rlabel))

    # add labels:
    #a.axis["top"].label.set_text(thlabel)
    #a.axis["left"].label.set_text(rlabel)

    # create a parasite axes whose transData is theta, r:
    auxa = a.get_aux_axes(tr)
    # make aux_ax to have a clip path as in a?:
    auxa.patch = a.patch 
    # this has a side effect that the patch is drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to prevent this:
    a.patch.zorder = -2

    # add sector lines for both dimensions:
#    thticks = grid_helper.grid_info['lon_info'][0]
#    rticks = grid_helper.grid_info['lat_info'][0]
 #   for th in thticks[1:-1]: # all but the first and last
#        auxa.plot([th, th], [r0, r1], '--', c='grey', zorder=-1)
  #  for ri, r in enumerate(rticks):
        # plot first r line as axes border in solid black only if it isn't at r=0
   #     if ri == 0 and r != 0:
     #       ls, lw, color = 'solid', 2, 'black'
    #    else:
      #      ls, lw, color = 'dashed', 1, 'grey'
        # From http://stackoverflow.com/a/19828753/2020363
       # auxa.add_artist(plt.Circle([0, 0], radius=r, ls=ls, lw=lw, color=color, fill=False,
        #                transform=auxa.transData._b, zorder=-1))
    return auxa



    
    
 
ie = 4.0
il = 3.18
ia = 0.276
ig=0.33

vis_angle = ia#0.31#np.mean(cc)
## POLAR PLOT OF RELATIVE POSITIONS
#BL = is approx 32 pixels
binn2=38 # distance bins
binn1=38

dr = 0.15 # width of distance bins
sr = 0.25 # start point of distance
maxr=sr+(dr*binn2)
#vis_angle = 0.31
theta2 = np.linspace(-ia,ia, binn1+1)
#theta2 = np.linspace(-vis_angle,vis_angle, binn1+1)
r2 = np.linspace(sr, maxr, binn2+1)
areas = pi*((r2+dr)**2-r2**2)

#aa = np.load( 'decay_exponent.npy')
#bb = np.load( 'interaction_length.npy')
#cc = np.load( 'interaction_angle.npy')
#ig=np.mean(aa)
#ir=np.mean(bb)
#areas=np.ones_like(r2)
areas = np.exp(-((r2/il)**ie))#**ig
areas[r2<ig]=0.0
    
#ig=3.11
#ir=3.87
#areas = np.exp(-r2/ir)*np.tanh(r2/ig)
areas=np.tile(areas,(binn1,1)).T

#for i in range(binn1):
#    for j in range(binn2):
#        if ((theta2[i])>vis_angle):
#            if (theta2[i])<2*pi-vis_angle:
#                areas[j,i]=-1;

hista2 =areas

size = 8
# make a square figure

fig1=plt.figure(figsize=(8,8))


dx, dy = 0.15, 0.15

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-3, 3 + dy, dy),                slice(-3, 3 + dx, dx)]
z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

#ax2 = fractional_polar_axes(fig1 ,thlim=(-degrees(ia), degrees(ia)),rlim=(sr,maxr),step=(degrees(2*ia/(binn1+1)),dr))
#ax2 = fractional_polar_axes(fig1 ,thlim=(-degrees(ia), degrees(ia)),rlim=(sr,maxr),step=(degrees(2*ia/(binn1+1)),dr))
# example spiral plot:
#thstep = 10
#th = np.arange(0, 180+thstep, thstep) # deg
#rstep = 1/(len(th)-1)
#r = np.arange(0, 1+rstep, rstep)
#a1.plot(th, r, 'b')
#f1.show()

ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=np.max(hista2),cmap='viridis')
#im=ax2.pcolormesh(hista2,vmin=0,vmax=np.max(hista2),cmap='viridis')
#im=ax2.pcolormesh(2*y,x,z,vmin=0,vmax=np.max(hista2),cmap='viridis')
#ax2.xaxis.set_visible(False)

# angle lines
ax2.set_thetagrids(angles=np.arange(0),labels=['','10', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
#ax1.yaxis.set_visible(False)
#ax1.set_ylim(0,0.5)
ax2.yaxis.set_visible(False)
#ax1.set_thetagrids(angles=np.arange(0,36,4.5),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
ax1.set_thetagrids(angles=np.degrees(np.arange(-0.4,0.5,0.2)),labels=['-0.4','-0.2', '0', '0.2', '0.4'],frac=1.1)
#ax1.set_rgrids(radii=np.arange(1,6,1),labels=['-0.4','-0.2', '0', '0.2', '0.4'],frac=1.1)
ax1.set_rgrids(radii=np.arange(1,6,0.9),labels=['1','2','3','4','5',''], angle=22.918)
#ax1.set_rgrids(radii=np.arange(1,6,0.1))
#colourbar
position=fig1.add_axes([1.1,0.22,0.02,0.5])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Social weighting', rotation=90,fontsize='large',labelpad=15)      

#body length legend - draws the ticks and 
axes=ax2            
factor = 0.98
d = axes.get_yticks()[-1] #* factor
r_tick_labels = [0] + axes.get_yticks()
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2

# fixed offsets in x
offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)

# apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)

# plot the 'tick labels'
for ii in range(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


fig1.savefig("rules.png",bbox_inches='tight',dpi=100)
# plot the 'spine'

    