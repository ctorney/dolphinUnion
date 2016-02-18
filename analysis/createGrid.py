import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'

    posDF = pd.read_csv(posfilename) 
    posDF['clip']=index
    posDF = posDF[posDF['frame']%60==0]
    posDF['x']=posDF['x']-min(posDF['x'])
    posDF['y']=posDF['y']-min(posDF['y'])
    xrange = max(posDF['x'])
    yrange = max(posDF['y'])
    nx = math.ceil(xrange/32)
    ny = math.ceil(yrange/32)
    grid = np.zeros((nx,ny,2))
    gridPos = np.zeros((nx,ny,2))
    xdirs = np.cos(posDF['heading'].values)
    ydirs = np.sin(posDF['heading'].values)
    xp = posDF['x'].values
    yp = posDF['y'].values
    kappa = 32*32
    for i in range(nx):
        for j in range(ny):
            gx = i * 32
            gy = j * 32
            dists = ((posDF['x'].values - gx)**2 + (posDF['y'].values - gy)**2)
            weights = np.exp(-dists/kappa)
            gridPos[i,j,0]=gx
            gridPos[i,j,1]=gy
            grid[i,j,0]=np.sum(weights*xdirs)/np.sum(weights)
            grid[i,j,1]=np.sum(weights*ydirs)/np.sum(weights)

    plt.quiver(xp,yp,xdirs,ydirs,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)
    plt.axis('equal')
    break

    

