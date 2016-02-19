import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.animation as ani

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

FFMpegWriter = ani.writers['ffmpeg']
metadata = dict(title='animation of movement')
writer = FFMpegWriter(fps=10, metadata=metadata)

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
    xh = np.cos(posDF['heading'].values)
    yh = np.sin(posDF['heading'].values)
    xdirs = posDF['dx'].values
    ydirs = posDF['dy'].values
    xp = posDF['x'].values
    yp = posDF['y'].values
    kappa = 32.0#*32.0
    for i in range(nx):
        for j in range(ny):
            gx = i * 32
            gy = j * 32
            dists = np.sqrt(((posDF['x'].values - gx)**2 + (posDF['y'].values - gy)**2))
            weights = np.exp(-dists/kappa)
            gridPos[i,j,0]=gx
            gridPos[i,j,1]=gy
            xav = np.sum(weights*xdirs)/np.sum(weights)
            yav = np.sum(weights*ydirs)/np.sum(weights)
            grid[i,j,0]=xav/math.sqrt(xav**2+yav**2)
            grid[i,j,1]=yav/math.sqrt(xav**2+yav**2)
            

    #plt.quiver(xp,yp,xh,yh,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    #plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)
    
    maxRange = 0
    flen = len(posDF.groupby('frame'))
    Xcentroids = np.zeros((flen,1))
    Ycentroids = np.zeros((flen,1))
    for fnum, frame in posDF.groupby('frame'):
        dist = max(frame['x'].values)-min(frame['x'].values)
        if dist>maxRange:
            maxRange=dist
        dist = max(frame['y'].values)-min(frame['y'].values)
        if dist>maxRange:
            maxRange=dist
        Xcentroids(fnum) = np.average(frame['x'].values)
        Ycentroids(fnum) = np.average(frame['y'].values)
     
    sz = math.ceil(maxRange/32)*16
    
    
    fig = plt.figure()#figsize=(10, 10), dpi=5)
    plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)
    l, = plt.plot([], [], 'ro')
    #plt.axis([0,4000, 2000,-2000])
    plt.axis('equal')

    
    seconds = 10
    totalFrames = 10*seconds
    fc = 0
    with writer.saving(fig, "move.mp4", totalFrames):# len(posDF.groupby('frame'))):


        for fnum, frame in posDF.groupby('frame'):
            fc = fc + 1
            if fc>totalFrames:
                break
            xp = frame['x'].values
            yp = frame['y'].values
            xc = np.average(xp)
            yc = np.average(yp)
            l.set_data(xp, yp)
            l.axes.set_xlim(xc-sz,xc+sz)
            l.axes.set_ylim(yc-sz,yc+sz)
    
            writer.grab_frame()
    break

    


