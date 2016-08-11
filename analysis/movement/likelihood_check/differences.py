
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy.stats
from scipy.stats import binned_statistic_2d
from scipy.ndimage.filters import gaussian_filter1d
import constantModelAlign
import decayModelAlign
import networkModelAlign
from scipy.interpolate import interp1d

plt.close('all')

cma = constantModelAlign.constantP()
dma = decayModelAlign.decayP()
nma = networkModelAlign.networkP()



    
    
    
neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')




nn1 = np.zeros_like(mvector)-1
nn2 = np.zeros_like(mvector)-1
nndiff = np.zeros_like(mvector)-1

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
        nndiff[i]=sndmin-np.min(rowLoc)


# decay model versus constant model        
dectocon=dma-cma
# decay model versus network model
dectonet=dma-nma


                            

rmin=1
rmax=20
nbins=19
bs = (rmax-rmin)/nbins


plt.figure()

# average
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
# standard error
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err/=np.sqrt(counts)


bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])


bmsmooth = gaussian_filter1d(bin_means,sigma=1.5*bs,mode='nearest')

f2 = interp1d(bin_centres, bmsmooth, kind='cubic')

newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)

plt.plot(newX, f2(newX), label='nma1')

bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])

err, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)


bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])


bmsmooth = gaussian_filter1d(bin_means,sigma=1.5*bs,mode='nearest')

f2 = interp1d(bin_centres, bmsmooth, kind='cubic')

newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)

plt.plot(newX, f2(newX), label='nma2')

#plt.ylim([-0.08,0.08])
plt.axhline(y=0, color='k')
plt.legend()
plt.figure()

rmin=1.0
rmax=10
nbins=20

bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])

err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)




bmsmooth = gaussian_filter1d(bin_means,sigma=1.5*bs,mode='nearest')

f2 = interp1d(bin_centres, bmsmooth, kind='cubic')

newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)

plt.plot(newX, f2(newX), label='cma1')
#plt.plot(bin_centres, bin_means, label='cma1')

#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o',label='cma1')

rmin=0
rmax=10
nbins=20

bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nndiff,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err, bin_edges, binnumber = scipy.stats.binned_statistic(nndiff,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nndiff,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)



bmsmooth = gaussian_filter1d(bin_means,sigma=1.5*bs,mode='nearest')

f2 = interp1d(bin_centres, bmsmooth, kind='cubic')

newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)

plt.plot(newX, f2(newX), label='cma2')
#plt.plot(bin_centres, bin_means, label='cma1')




#plt.ylim([-0.08,0.08])
plt.axhline(y=0, color='k')
plt.legend()

#
#plt.figure()
#
#
##metrics1=metrics1[metrics1>=0]
#like_diff2=like_diff[metrics2>=0]
##metrics2=metrics2[metrics2>=0]
#
#
#rmin=0
#rmax=10
#nbins=10
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
#
#
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
#
#
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
##plt.figure()
#
##bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff2,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
##
##err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff2,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
##counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff2,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
##bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
##err/=np.sqrt(counts)
##
##
##plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
##
###plt.ylim([-0.08,0.08])
##plt.axhline(y=0, color='k')
##
##
##plt.figure()
#metrics1 = np.zeros_like(mvector)-1
#metrics2 = np.zeros_like(mvector)-1
#
#for i,db in enumerate(neighbours):
#    rowLoc = np.zeros((0)).astype(np.float32)
#    rowAng = np.zeros((0)).astype(np.float32)
#    for row in db:
#        if row[0]>0 and row[0]<10.0 and row[1]>-0.28 and row[1]<0.28:
#            
#            rowLoc = np.append(rowLoc,row[0])
#
#            rowAng = np.append(rowAng,row[1])
#
#    
#        
#    if len(rowLoc)>0:
#        fmin = np.min(rowLoc)#-np.min(rowLoc)
#        #sndmin = np.partition(np.reshape(rowLoc,(len(rowLoc))),1)[1]
#        i1 = np.argwhere(rowLoc==fmin)
#    #i2 = np.argwhere(rowLoc==sndmin)
#        
#        metrics1[i] = len(rowLoc)#abs(rowAng[i1]-np.mean(rowAng[np.arange(len(rowAng))!=i1[0][0]]))#len(rowLoc)# np.max(abs(rowAng[i1]-rowAng))
#        #metrics1[i] = abs(rowAng[i2[0][0]]) #- abs(rowAng[i2[0][0]])#np.mean(rowAng[np.arange(len(rowAng))!=i1[0][0]]))#rowAng[i2[0][0]])#abs(np.max(rowAng)-np.min(rowAng)) #np.std(rowAng)#abs(rowAng[i1]-np.mean(rowAng))
#    if len(rowLoc)>=1:
#        fmin = np.min(rowLoc)#-np.min(rowLoc)
#        #sndmin = np.partition(np.reshape(rowLoc,(len(rowLoc))),1)[1]
#        i1 = np.argwhere(rowLoc==fmin)
#        #i2 = np.argwhere(rowLoc==sndmin)
#        
#        #metrics2[i] = abs(rowAng[i1]-np.mean(rowAng[np.arange(len(rowAng))!=i1[0][0]]))#len(rowLoc)# np.max(abs(rowAng[i1]-rowAng))
#        #metrics2[i] = abs(rowAng[i1[0][0]]) #- abs(rowAng[i2[0][0]])#np.mean(rowAng[np.arange(len(rowAng))!=i1[0][0]]))#rowAng[i2[0][0]])#abs(np.max(rowAng)-np.min(rowAng)) #np.std(rowAng)#abs(rowAng[i1]-np.mean(rowAng))
#        metrics2[i] = abs(rowAng[i1])#-np.mean(rowAng))
#        
#
#    
#like_diff=dma-nma
#
#np.mean(metrics1[np.bitwise_and(like_diff>np.mean(like_diff),metrics1>-1)])
#
#like_diff1=like_diff[metrics1>=0]
##metrics1=metrics1[metrics1>=0]
#
#plt.figure()
#rmin=0
#rmax=0.2
#nbins=10
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,like_diff,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,like_diff,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,like_diff,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics2,like_diff,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o')
#
#plt.figure()
#
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,nma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o',label='nma')
#
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,cma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o',label='cma')
#
#
#bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic='mean', bins=nbins, range=[rmin,rmax])
#
#
#
#err, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic=np.std, bins=nbins, range=[rmin,rmax])
#counts, bin_edges, binnumber = scipy.stats.binned_statistic(metrics1,dma,   statistic='count', bins=nbins,range=[rmin,rmax])
#bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
#err/=np.sqrt(counts)
#
#
#plt.errorbar(bin_centres, bin_means, yerr=err, fmt='-o',label='dma')
#plt.legend()
##plt.ylim([-0.2,0.2])
##plt.axhline(y=0, color='k')
##plt.figure()
#
#
#
##
##
##decayBetter = neighbours[dma<cma]
##
##dbmv = mvector[dma<cma]
##
##locations = np.zeros((0,2)).astype(np.float32)
##
##for i,db in enumerate(decayBetter):
##    rowLoc = np.zeros((0,2)).astype(np.float32)
##    for row in db:
##        if row[0]>0:
##            rowLoc = np.vstack((rowLoc,[row[0],row[1]]))
##    locations = np.vstack((locations,rowLoc))
##
##
##
#### POLAR PLOT OF RELATIVE POSITIONS
##binn2=19 # distance bins
##binn1=72
##dr = 0.3 # width of distance bins
##sr = 0.25 # start point of distance
##maxr=sr+(dr*binn2)
##theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
##r2 = np.linspace(sr, maxr, binn2+1)
##areas = pi*((r2+dr)**2-r2**2)/binn1
##areas = areas[0:-1]
##areas=np.tile(areas,(binn1,1)).T
##locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi 
##hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
##hista2 = (10*hista2/areas)/45692 # divide by number of individuals 
##
##
###fig = plt.figure(figsize=(18, 12))
##ax2=plt.subplot(projection="polar",frameon=False)
##
##
##
##im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
##plt.savefig("dmaltcma.png",bbox_inches='tight',dpi=100)
##
##plt.figure()
##decayBetter = neighbours[dma>cma]
##
##dbmv = mvector[dma>cma]
##
##locations = np.zeros((0,2)).astype(np.float32)
##
##for i,db in enumerate(decayBetter):
##    rowLoc = np.zeros((0,2)).astype(np.float32)
##    for row in db:
##        if row[0]>0:
##            rowLoc = np.vstack((rowLoc,[row[0],row[1]]))
##    locations = np.vstack((locations,rowLoc))
##
##
##
#### POLAR PLOT OF RELATIVE POSITIONS
##binn2=19 # distance bins
##binn1=72
##dr = 0.3 # width of distance bins
##sr = 0.25 # start point of distance
##maxr=sr+(dr*binn2)
##theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
##r2 = np.linspace(sr, maxr, binn2+1)
##areas = pi*((r2+dr)**2-r2**2)/binn1
##areas = areas[0:-1]
##areas=np.tile(areas,(binn1,1)).T
##locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi 
##hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
##hista2 = (10*hista2/areas)/45692 # divide by number of individuals 
##
##
###fig = plt.figure(figsize=(18, 12))
##ax2=plt.subplot(projection="polar",frameon=False)
##
##
##
##im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
##
##plt.savefig("dmagtcma.png",bbox_inches='tight',dpi=100)
##
##plt.figure()
##
##decayBetter = neighbours[dma<nma]
##
##dbmv = mvector[dma<nma]
##
##locations = np.zeros((0,2)).astype(np.float32)
##
##for i,db in enumerate(decayBetter):
##    rowLoc = np.zeros((0,2)).astype(np.float32)
##    for row in db:
##        if row[0]>0:
##            rowLoc = np.vstack((rowLoc,[row[0],row[1]-dbmv[i]]))
##    locations = np.vstack((locations,rowLoc))
##
##
##
#### POLAR PLOT OF RELATIVE POSITIONS
##binn2=19 # distance bins
##binn1=72
##dr = 0.3 # width of distance bins
##sr = 0.25 # start point of distance
##maxr=sr+(dr*binn2)
##theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
##r2 = np.linspace(sr, maxr, binn2+1)
##areas = pi*((r2+dr)**2-r2**2)/binn1
##areas = areas[0:-1]
##areas=np.tile(areas,(binn1,1)).T
##locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi 
##hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
##hista2 = (10*hista2/areas)/45692 # divide by number of individuals 
##
##
###fig = plt.figure(figsize=(18, 12))
##ax2=plt.subplot(projection="polar",frameon=False)
##
##
##
##im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
##plt.savefig("dmaltnma.png",bbox_inches='tight',dpi=100)
##
##plt.figure()
##decayBetter = neighbours[dma>nma]
##
##dbmv = mvector[dma>nma]
##
##locations = np.zeros((0,2)).astype(np.float32)
##
##for i,db in enumerate(decayBetter):
##    rowLoc = np.zeros((0,2)).astype(np.float32)
##    for row in db:
##        if row[0]>0:
##            rowLoc = np.vstack((rowLoc,[row[0],row[1]-dbmv[i]]))
##    locations = np.vstack((locations,rowLoc))
##
##
##
#### POLAR PLOT OF RELATIVE POSITIONS
##binn2=19 # distance bins
##binn1=72
##dr = 0.3 # width of distance bins
##sr = 0.25 # start point of distance
##maxr=sr+(dr*binn2)
##theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
##r2 = np.linspace(sr, maxr, binn2+1)
##areas = pi*((r2+dr)**2-r2**2)/binn1
##areas = areas[0:-1]
##areas=np.tile(areas,(binn1,1)).T
##locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi 
##hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
##hista2 = (10*hista2/areas)/45692 # divide by number of individuals 
##
##
###fig = plt.figure(figsize=(18, 12))
##ax2=plt.subplot(projection="polar",frameon=False)
##
##
##
##im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
##
##plt.savefig("dmagtnma.png",bbox_inches='tight',dpi=100)