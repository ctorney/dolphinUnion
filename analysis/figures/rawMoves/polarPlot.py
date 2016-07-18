import os
import csv
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import pandas as pd
import math
from matplotlib import transforms
from viridis import viridis
plt.register_cmap(name='viridis', cmap=viridis)
viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
plt.register_cmap(name='viridis_r', cmap=viridis_r)

 

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

locations=np.load('locations.npy')

## POLAR PLOT OF RELATIVE POSITIONS
#BL = is approx 32 pixels
binn2=19 # distance bins
binn1=72

dr = 0.5 # width of distance bins
sr = 0.25 # start point of distance
maxr=sr+(dr*binn2)
theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
r2 = np.linspace(sr, maxr, binn2+1)
areas = pi*((r2+dr)**2-r2**2)
areas = areas[0:-1]
areas=np.tile(areas,(binn1,1)).T

# wrap to [0, 2pi]
locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi

hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=1)[0]  

hista2 = 100*hista2/areas
#probability per unit area x10-3
size = 8
# make a square figure

fig1=plt.figure(figsize=(6,6))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,hista2,lw=0.0,vmin=0,vmax=1.0,cmap='viridis')
#np.min(hista2)
np.max(hista2)
ax2.yaxis.set_visible(False)

# angle lines
ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
#colourbar
position=fig1.add_axes([1.1,0.12,0.04,0.8])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Neighbour density (x0.01)', rotation=90,fontsize='x-large',labelpad=15)      

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
for ii in xrange(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


fig1.savefig("densityHM.tiff",bbox_inches='tight',dpi=200)
# plot the 'spine'


       
      
## POLAR PLOT OF ALIGNMENT
cosRelativeAngles = np.cos(locations[:,2])
sinRelativeAngles = np.sin(locations[:,2])

# find the average cos and sin of the relative headings to calculate circular statistics
histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  

# mean is atan and std dev is 1-R
relativeAngles = np.arctan2(histsin,histcos)
stdRelativeAngles = np.sqrt( 1 - np.sqrt(histcos**2+histsin**2))


fig1=plt.figure(figsize=(6,6))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,stdRelativeAngles,lw=0.0,vmin=np.min(stdRelativeAngles),vmax=np.max(stdRelativeAngles),cmap='viridis')
#im=ax2.pcolormesh(theta2,r2,stdRelativeAngles,lw=0.0,vmin=0,vmax=0.8,cmap='viridis_r')
ax2.yaxis.set_visible(False)

# angle lines
ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
#colourbar
position=fig1.add_axes([1.1,0.12,0.04,0.8])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Circular variance', rotation=90,fontsize='x-large',labelpad=15)      

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
for ii in xrange(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


fig1.savefig("order.tiff",bbox_inches='tight',dpi=200)


## POLAR PLOT OF ATTRACTION


# find the average cos and sin of the relative headings to calculate circular statistics
histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  


angles = 0.5*(theta2[0:-1]+theta2[1:])
angles=np.tile(angles,(binn2,1))

toOrigin = -(histcos*np.cos(angles) + histsin*np.sin(angles))
fig1=plt.figure(figsize=(6,6))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,toOrigin,lw=0.0,vmin=np.min(toOrigin),vmax=np.max(toOrigin),cmap='viridis')
ax2.yaxis.set_visible(False)

# angle lines
ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
#colourbar
position=fig1.add_axes([1.1,0.12,0.04,0.8])
cbar=plt.colorbar(im,cax=position) 
cbar.set_label('Attraction', rotation=90,fontsize='x-large',labelpad=15)      

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
for ii in xrange(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


fig1.savefig("toOrigin.tiff",bbox_inches='tight',dpi=200)

### POLAR PLOT OF ATTRACTION FORCE
#
#
#relAccX = np.cos(locations[:,3])*locations[:,4]
#relAccY = np.sin(locations[:,3])*locations[:,4]
#
#
## find the average cos and sin of the relative headings to calculate circular statistics
#histX=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=relAccX, statistic='mean', bins=[r2,theta2])[0]  
#histY=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=relAccY, statistic='mean', bins=[r2,theta2])[0]  
#
#
#angles = 0.5*(theta2[0:-1]+theta2[1:])
#angles=np.tile(angles,(binn2,1))
#
#toOrigin = -(histX*np.cos(angles) + histY*np.sin(angles))
#fig1=plt.figure(figsize=(8,8))
#ax2=plt.subplot(projection="polar",frameon=False)
#im=ax2.pcolormesh(theta2,r2,toOrigin,lw=0.0,vmin=np.min(toOrigin),vmax=np.max(toOrigin),cmap='viridis')
#ax2.yaxis.set_visible(False)
#
## angle lines
#ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
#ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
#ax1.yaxis.set_visible(False)
#ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
##colourbar
#position=fig1.add_axes([1.1,0.12,0.04,0.8])
#cbar=plt.colorbar(im,cax=position) 
#cbar.set_label('Attraction', rotation=90,fontsize='xx-large',labelpad=15)      
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
#for ii in xrange(len(r_ticks)):
#    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
#axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
#
#
#fig1.savefig("accToOrigin.png",bbox_inches='tight',dpi=100)
