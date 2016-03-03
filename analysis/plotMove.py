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
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
DATADIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'
df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    gridfilename = DATADIR + '/GRID_' + str(index) + '_' + noext + '.npy'
    gridPosfilename = DATADIR + '/GRIDPOS_' + str(index) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
    
    posDF = posDF[posDF['frame']%60==0]
    
#    posDF['x']=posDF['x']-min(posDF['x'])
#    posDF['y']=posDF['y']-min(posDF['y'])
#    xrange = max(posDF['x'])
#    yrange = max(posDF['y'])
#    nx = math.ceil(xrange/32)
#    ny = math.ceil(yrange/32)
#    grid = np.zeros((nx,ny,2))
#    gridPos = np.zeros((nx,ny,2))
#    xh = np.cos(posDF['heading'].values)
#    yh = np.sin(posDF['heading'].values)
#    xdirs = posDF['dx'].values
#    ydirs = posDF['dy'].values
#    xp = posDF['x'].values
#    yp = posDF['y'].values
#    kappa = 32.0*32.0
#    for i in range(nx):
#        for j in range(ny):
#            gx = i * 32
#            gy = j * 32
#            dists = (((posDF['x'].values - gx)**2 + (posDF['y'].values - gy)**2))
#            weights = np.exp(-dists/kappa)
#            gridPos[i,j,0]=gx
#            gridPos[i,j,1]=gy
#            xav = np.sum(weights*xdirs)/np.sum(weights)
#            yav = np.sum(weights*ydirs)/np.sum(weights)
#            grid[i,j,0]=xav/math.sqrt(xav**2+yav**2)
#            grid[i,j,1]=yav/math.sqrt(xav**2+yav**2)
            
    grid = np.load(gridfilename)
    gridPos = np.load(gridPosfilename)
    #plt.quiver(xp,yp,xh,yh,angles='xy', scale_units='xy', color='r', scale=1.0/32.0)
    #plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0)
    
    winLen = 30
    w = np.kaiser(winLen,1)
    w = w/w.sum()
    maxRange = 0
    flen = len(posDF.groupby('frame'))
    Xcentroids = np.zeros((flen))
    Ycentroids = np.zeros((flen))
    fc=0
    for fnum, frame in posDF.groupby('frame'):
        dist = max(frame['x'].values)-min(frame['x'].values)
        if dist>maxRange:
            maxRange=dist
        dist = max(frame['y'].values)-min(frame['y'].values)
        if dist>maxRange:
            maxRange=dist
        Xcentroids[fc] = np.average(frame['x'].values)
        Ycentroids[fc] = np.average(frame['y'].values)
        fc=fc+1
    Xcentroids = np.r_[np.ones((winLen))*Xcentroids[0],Xcentroids,np.ones((winLen))*Xcentroids[-1]]
    Xcentroids = np.convolve(w/w.sum(),Xcentroids,mode='same')[(winLen):-(winLen)]
    
    Ycentroids = np.r_[np.ones((winLen))*Ycentroids[0],Ycentroids,np.ones((winLen))*Ycentroids[-1]]
    Ycentroids = np.convolve(w/w.sum(),Ycentroids,mode='same')[(winLen):-(winLen)]
    sz = math.ceil(maxRange/32)*16

    
    fig = plt.figure()#figsize=(10, 10), dpi=5)
    
    
    
    totalFrames =500
    fc = 0
    #with writer.saving(fig, "move.mp4", totalFrames):# len(posDF.groupby('frame'))):


    for fnum, frame in posDF.groupby('frame'):
        fc = fc + 1
        if fc>totalFrames:
            break
        xp = frame['x'].values
        yp = frame['y'].values
        xh = 0.1*frame['dx'].values
        yh = 0.1*frame['dy'].values
        xc = Xcentroids[fc]
        yc = Ycentroids[fc]
        plt.clf()
        plt.quiver(gridPos[:,:,0],gridPos[:,:,1],grid[:,:,0],grid[:,:,1],angles='xy', scale_units='xy', scale=1.0/32.0, headwidth=1)
        l, = plt.plot(xp,yp, 'ro')
        plt.quiver(xp,yp,xh,yh,angles='xy', scale_units='xy', color='r', scale=1.0/32.0, headwidth=1.5)
    #plt.axis([0,4000, 2000,-2000])
        plt.axis('equal')
        l.axes.get_xaxis().set_visible(False)
        l.axes.get_yaxis().set_visible(False)
        l.set_data(xp, yp)
        l.axes.set_xlim(xc-sz,xc+sz)
        l.axes.set_ylim(yc-sz,yc+sz)
        plt.savefig('frames/fig'+'{0:05d}'.format(fc)+'.png')
    
            #writer.grab_frame()
    break

    


