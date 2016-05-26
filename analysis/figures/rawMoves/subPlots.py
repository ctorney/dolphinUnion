import numpy as np
import matplotlib.pyplot as plt
import math
import os
from math import pi
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from viridis import viridis
from scipy.stats import binned_statistic_2d
from scipy.stats import norm

plt.register_cmap(name='viridis', cmap=viridis)
viridis_r = matplotlib.colors.LinearSegmentedColormap( 'viridis_r', matplotlib.cm.revcmap(viridis._segmentdata))
plt.register_cmap(name='viridis_r', cmap=viridis_r)
plt.close('all')


fig = plt.figure(figsize=(18, 12))
pax1 = plt.subplot2grid((2,4), (0,0), colspan=2,projection="polar",frameon=False)
pax2 = plt.subplot2grid((2,4), (0,2), colspan=2,projection="polar",frameon=False)
pax3 = plt.subplot2grid((2,4), (1, 0), colspan=2,projection="polar",frameon=False)
pax4 = plt.subplot2grid((2,4), (1, 2))
pax5 = plt.subplot2grid((2,4), (1, 3))

plt.tight_layout(pad=2.5, w_pad=3.0, h_pad=2.50)


locations=np.load('locations.npy')



## POLAR PLOT OF RELATIVE POSITIONS
binn2=19 # distance bins
binn1=72
dr = 0.3 # width of distance bins
sr = 0.25 # start point of distance
maxr=sr+(dr*binn2)
theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
r2 = np.linspace(sr, maxr, binn2+1)
areas = pi*((r2+dr)**2-r2**2)/binn1
areas = areas[0:-1]
areas=np.tile(areas,(binn1,1)).T
locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi
hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
hista2 = (10*hista2/areas)/45692 # divide by number of individuals 

im=pax1.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=1.0,cmap='viridis')

pax1.yaxis.set_visible(False)
pax1.set_thetagrids(angles=np.arange(0,360,45),labels=['', '', '', '', '', '','', ''],frac=1.1)
position=fig.add_axes([pax1._position.bounds[0]+0.39,pax1._position.bounds[1]+0.0,0.015,0.4])
cbar=plt.colorbar(im,ax=pax1,cax=position) 
cbar.set_label('Neighbour density (x0.1)', rotation=90,fontsize='x-large',labelpad=15)  
#plt.show()
plt.savefig("tmp.png",bbox_inches='tight',dpi=300)
ax1 = pax1.figure.add_axes(pax1.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=pax1.get_theta_direction(), theta_offset=pax1.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '45°', '90°', '135°', 'back', '225°','270°', '315°'],frac=1.1)
##body length legend - draws the ticks and 
axes=pax1            
factor = 0.98
d = axes.get_yticks()[-1] #* factor
r_tick_labels = [0] + axes.get_yticks()
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
offset_spine = transforms.ScaledTranslation(-140, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-50, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-100, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
#
## plot the 'tick labels'
for ii in range(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


pax1.text(-0.2, 1.1, 'A', transform=pax1.transAxes, fontsize=24, va='top', ha='right')


###################################################
###################################################
################SECOND PLOT########################
###################################################
###################################################



## POLAR PLOT OF ALIGNMENT
cosRelativeAngles = np.cos(locations[:,2])
sinRelativeAngles = np.sin(locations[:,2])

# find the average cos and sin of the relative headings to calculate circular statistics
histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  

# mean is atan and std dev is 1-R
relativeAngles = np.arctan2(histsin,histcos)
stdRelativeAngles = np.sqrt( 1 - np.sqrt(histcos**2+histsin**2))


#fig1=plt.figure(figsize=(6,6))


#im=ax2.pcolormesh(theta2,r2,stdRelativeAngles,lw=0.0,vmin=np.min(stdRelativeAngles),vmax=np.max(stdRelativeAngles),cmap='viridis')
im=pax2.pcolormesh(theta2,r2,stdRelativeAngles,lw=0.0,vmin=0.1,vmax=0.6,cmap='viridis')

pax2.yaxis.set_visible(False)
pax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '', '', '', '', '','', ''],frac=1.1)
position2=fig.add_axes([pax2._position.bounds[0]+0.295,pax2._position.bounds[1]+0.0,0.015,0.4])
cbar=plt.colorbar(im,ax=pax2,cax=position2) 
cbar.set_label('Circular variance', rotation=90,fontsize='x-large',labelpad=15) 
#plt.show()
plt.savefig("tmp.png",bbox_inches='tight',dpi=300)
ax2 = pax2.figure.add_axes(pax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=pax2.get_theta_direction(), theta_offset=pax2.get_theta_offset())
ax2.yaxis.set_visible(False)
ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '45°', '90°', '135°', 'back', '225°','270°', '315°'],frac=1.1)
##body length legend - draws the ticks and 
axes=pax2            
factor = 0.98
d = axes.get_yticks()[-1] #* factor
r_tick_labels = [0] + axes.get_yticks()
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
offset_spine = transforms.ScaledTranslation(-140, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-50, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-100, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
#
## plot the 'tick labels'
for ii in range(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


pax2.text(-0.2, 1.1, 'B', transform=pax2.transAxes, fontsize=24, va='top', ha='right')


###################################################
###################################################
################THIRD PLOT#########################
###################################################
###################################################



# find the average cos and sin of the relative headings to calculate circular statistics
histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  


angles = 0.5*(theta2[0:-1]+theta2[1:])
angles=np.tile(angles,(binn2,1))

toOrigin = -(histcos*np.cos(angles) + histsin*np.sin(angles))
#fig1=plt.figure(figsize=(6,6))

im=pax3.pcolormesh(theta2,r2,toOrigin,lw=0.0,vmin=-1,vmax=1,cmap='viridis')

pax3.yaxis.set_visible(False)
pax3.set_thetagrids(angles=np.arange(0,360,45),labels=['', '', '', '', '', '','', ''],frac=1.1)
position3=fig.add_axes([pax3._position.bounds[0]+0.295,pax3._position.bounds[1]+0.0,0.015,0.4])
cbar=plt.colorbar(im,ax=pax3,cax=position3) 
cbar.set_label('Relative heading', rotation=90,fontsize='x-large',labelpad=15) 
#plt.show()
plt.savefig("tmp.png",bbox_inches='tight',dpi=300)
ax2 = pax3.figure.add_axes(pax3.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=pax3.get_theta_direction(), theta_offset=pax3.get_theta_offset())
ax2.yaxis.set_visible(False)
ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '45°', '90°', '135°', 'back', '225°','270°', '315°'],frac=1.1)
##body length legend - draws the ticks and 
axes=pax3            
factor = 0.98
d = axes.get_yticks()[-1] #* factor
r_tick_labels = [0] + axes.get_yticks()
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
offset_spine = transforms.ScaledTranslation(-140, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-50, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-100, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
#
## plot the 'tick labels'
for ii in range(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='x-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')


pax3.text(-0.2, 1.1, 'C', transform=pax3.transAxes, fontsize=24, va='top', ha='right')


###################################################
###################################################
################FINAL PLOTS########################
###################################################
###################################################

HD = os.getenv('HOME')
MOVEFILE = HD + '/workspace/dolphinUnion/analysis/movement/pdata/mvector.npy'
moves = np.load(MOVEFILE)

#fit to centre
xs=0.05
c1 = np.sum((moves>-xs)&(moves<xs))
p1 = c1/len(moves)
pdf1 = p1/(2.0*xs)

sigma = 1.0/(pdf1*math.sqrt(2*math.pi))

xs=0.6
xp = np.linspace(-xs, xs, 1000)
gpdf = norm.pdf(xp,np.mean(moves),sigma)
pax4.plot(xp, gpdf,'-',linewidth=3,color="slategrey")
pax4.fill_between(xp,np.zeros_like(xp), gpdf,alpha=0.4)#, fc='#AAAAFF')
pax4.hist(moves,normed=True,range=[-xs,xs],bins=40,color="midnightblue")
pax4.set_xlim(-xs,xs)


#pax4.plot(np.arange(0,1,0.1),np.sin(np.arange(0,1,0.1)))
#pax4.hist(moves,range=[0.1,0.1],bins=100)
pax4.text(-0.2, 1.1, 'D', transform=pax4.transAxes, fontsize=24, va='top', ha='right')
pax4.set_xlabel('Turn angle',fontsize=16)
pax4.set_ylabel('Probability density',fontsize=16)



pax5.plot(np.arange(0,1,0.1),np.sin(np.arange(0,1,0.1)))
pax5.text(-0.2, 1.1, 'E', transform=pax5.transAxes, fontsize=24, va='top', ha='right')

pax5.set_xlabel('test',fontsize=20)
pax5.set_ylabel('test',fontsize=20)


plt.show()
plt.savefig("all.png",bbox_inches='tight',dpi=300)
