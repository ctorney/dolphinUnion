
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy.stats
import constantModelAlign
import decayModelAlign
import networkModelAlign

cma = constantModelAlign.constantP()
dma = decayModelAlign.decayP()
nma = networkModelAlign.networkP()

#plt.plot(cma,dma,'.')
#plt.figure()
#plt.hist(cma-dma,bins=100,range=[-0.1,0.1])

np.mean(cma)
np.mean(dma)



    
    
    
neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')




metrics = np.zeros_like(mvector)

for i,db in enumerate(neighbours):
    rowLoc = np.zeros((0,1)).astype(np.float32)
    for row in db:
        if row[0]>0 and row[0]<10.0 and row[1]>-0.3 and row[1]<0.3:
            
            rowLoc = np.vstack((rowLoc,row[0]))
    if len(rowLoc):
        metrics[i]=np.min(rowLoc)#-np.min(rowLoc)
        

    
like_diff=dma-cma
like_diff=like_diff[metrics!=0]
metrics=metrics[metrics!=0]
np.average(metrics[like_diff>0])
np.average(metrics[like_diff<0])

bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(metrics,like_diff,   statistic='mean', bins=20)#, range=[0,0.1])

plt.plot( 0.5*(bin_edges[:-1]+bin_edges[1:]),bin_means)

decayBetter = neighbours[dma<cma]

dbmv = mvector[dma<cma]

locations = np.zeros((0,2)).astype(np.float32)

for i,db in enumerate(decayBetter):
    rowLoc = np.zeros((0,2)).astype(np.float32)
    for row in db:
        if row[0]>0:
            rowLoc = np.vstack((rowLoc,[row[0],row[1]]))
    locations = np.vstack((locations,rowLoc))



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


#fig = plt.figure(figsize=(18, 12))
ax2=plt.subplot(projection="polar",frameon=False)



im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
plt.savefig("dmaltcma.png",bbox_inches='tight',dpi=100)

plt.figure()
decayBetter = neighbours[dma>cma]

dbmv = mvector[dma>cma]

locations = np.zeros((0,2)).astype(np.float32)

for i,db in enumerate(decayBetter):
    rowLoc = np.zeros((0,2)).astype(np.float32)
    for row in db:
        if row[0]>0:
            rowLoc = np.vstack((rowLoc,[row[0],row[1]]))
    locations = np.vstack((locations,rowLoc))



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


#fig = plt.figure(figsize=(18, 12))
ax2=plt.subplot(projection="polar",frameon=False)



im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')

plt.savefig("dmagtcma.png",bbox_inches='tight',dpi=100)

plt.figure()

decayBetter = neighbours[dma<nma]

dbmv = mvector[dma<nma]

locations = np.zeros((0,2)).astype(np.float32)

for i,db in enumerate(decayBetter):
    rowLoc = np.zeros((0,2)).astype(np.float32)
    for row in db:
        if row[0]>0:
            rowLoc = np.vstack((rowLoc,[row[0],row[1]-dbmv[i]]))
    locations = np.vstack((locations,rowLoc))



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


#fig = plt.figure(figsize=(18, 12))
ax2=plt.subplot(projection="polar",frameon=False)



im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')
plt.savefig("dmaltnma.png",bbox_inches='tight',dpi=100)

plt.figure()
decayBetter = neighbours[dma>nma]

dbmv = mvector[dma>nma]

locations = np.zeros((0,2)).astype(np.float32)

for i,db in enumerate(decayBetter):
    rowLoc = np.zeros((0,2)).astype(np.float32)
    for row in db:
        if row[0]>0:
            rowLoc = np.vstack((rowLoc,[row[0],row[1]-dbmv[i]]))
    locations = np.vstack((locations,rowLoc))



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


#fig = plt.figure(figsize=(18, 12))
ax2=plt.subplot(projection="polar",frameon=False)



im=ax2.pcolormesh(theta2,r2,hista2,lw=1.0,vmin=0,vmax=0.4,cmap='viridis')

plt.savefig("dmagtnma.png",bbox_inches='tight',dpi=100)