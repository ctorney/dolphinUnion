
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

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (214, 39, 40), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)  

    
    
    
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
        nndiff[i]=sndmin#-np.min(rowLoc)


# decay model versus constant model        
dectocon=dma-cma
# decay model versus network model
dectonet=dma-nma


                            

rmin=1.5
rmax=20.5
nbins=19
bs = (rmax-rmin)/nbins


### network model comparison

plt.figure()

## nearest neighbour
# average
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
# standard error
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err/=np.sqrt(counts)

# smoothing
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

plt.plot(newX, f2(newX), label='1st neighbor',color=tableau20[0],linewidth=1.5)
plt.fill_between(newX, uf2(newX),lf2(newX),facecolor=tableau20[0],linewidth=0, alpha=0.25)


## 2nd nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])

err, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectonet,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)


bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])


bmsmooth = gaussian_filter1d(bin_means,sigma=2.0*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')


plt.plot(newX, f2(newX), label='2nd neighbor',color=tableau20[1])
plt.fill_between(newX, uf2(newX),lf2(newX),facecolor=tableau20[1],linewidth=0, alpha=0.1)


plt.axhline(y=0, color='k')
plt.xlabel('distance')
plt.ylabel('model difference')
plt.legend()
plt.xlim([2,20])
plt.savefig('network_comparison.png')


## constant model comparison
plt.figure()

## nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn1,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)


#smoothing
bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

plt.plot(newX, f2(newX),color=tableau20[0], label='1st neighbor')
plt.fill_between(newX, uf2(newX),lf2(newX),facecolor=tableau20[0],linewidth=0, alpha=0.25)

## 2nd nearest neighbour
bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic='mean', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
err, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic=np.std, bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
counts, bin_edges, binnumber = scipy.stats.binned_statistic(nn2,dectocon,   statistic='count', bins=nbins,range=[rmin,rmax])#, range=[0,0.1])
bin_centres = 0.5*(bin_edges[:-1]+bin_edges[1:])
err/=np.sqrt(counts)

bmsmooth = gaussian_filter1d(bin_means,sigma=2*bs,mode='nearest')
f2 = interp1d(bin_centres, bmsmooth, kind='cubic')
newX = np.linspace(bin_centres[0], bin_centres[-1], num=51, endpoint=True)
errs = gaussian_filter1d(err,sigma=2*bs,mode='nearest')
uf2 = interp1d(bin_centres, bmsmooth+errs, kind='cubic')
lf2 = interp1d(bin_centres, bmsmooth-errs, kind='cubic')

plt.plot(newX, f2(newX),color=tableau20[1], label='2nd neighbor')
plt.fill_between(newX, uf2(newX),lf2(newX),facecolor=tableau20[1],linewidth=0, alpha=0.1)


#plt.ylim([-0.08,0.08])
plt.axhline(y=0, color='k')
plt.xlabel('distance')
plt.ylabel('model difference')
plt.legend()
plt.xlim([2,20])
plt.savefig('constant_comparison.png')
