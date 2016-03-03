import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

HD = os.getenv('HOME')

FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
DATADIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'
#first-pass barnes interpolation with decay length set to 32px
df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    gridfilename = DATADIR + '/GRID_' + str(index) + '_' + noext + '.npy'
    gridPosfilename = DATADIR + '/GRIDPOS_' + str(index) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
   
    posDF = posDF[posDF['frame']%60==0]
    #posDF['x']=posDF['x']#-min(posDF['x'])
    #posDF['y']=posDF['y']#-min(posDF['y'])
    xrange = max(posDF['x'])-min(posDF['x'])
    yrange = max(posDF['y'])-min(posDF['y'])
    minx = math.floor(min(posDF['x']))
    miny = math.floor(min(posDF['y']))
    nx = math.ceil(xrange/32)
    ny = math.ceil(yrange/32)
    grid = np.zeros((nx,ny,2))
    gridPos = np.zeros((nx,ny,2))
    xdirs = posDF['dx'].values
    ydirs = posDF['dy'].values
    xp = posDF['x'].values
    yp = posDF['y'].values
    kappa = 32.0*32.0
    for i in range(nx):
        for j in range(ny):
            gx = (i * 32) + minx
            gy = (j * 32) + miny
            dists = (((xp - gx)**2 + (yp - gy)**2))
            weights = np.exp(-dists/kappa)
            gridPos[i,j,0]= gx
            gridPos[i,j,1]= gy
            xav = np.sum(weights*xdirs)/np.sum(weights)
            yav = np.sum(weights*ydirs)/np.sum(weights)
            grid[i,j,0]=xav/math.sqrt(xav**2+yav**2)
            grid[i,j,1]=yav/math.sqrt(xav**2+yav**2)
    
    np.save(gridfilename, grid)
    np.save(gridPosfilename, gridPos)
    #plt.quiver(xp,yp,xdirs,ydirs,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    #plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)
    #plt.axis('equal')
    #break

    

