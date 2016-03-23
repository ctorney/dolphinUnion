
import os
import csv
import math
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from math import *


HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

df = pd.read_csv(FILELIST)
allDF = pd.DataFrame()
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    gridfilename = DATADIR + '/GRID_' + str(index) + '_' + noext + '.npy'
    gridPosfilename = DATADIR + '/GRIDPOS_' + str(index) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
    posDF['clip']=index
    posDF = posDF[posDF['frame']%120==0]
    dt = 120 # 60 frames is 1 second
    posDF['dtheta']=np.NaN
    posDF['env_heading']=np.NaN
    for index, row in posDF.iterrows():
        thisFrame =  row['frame']
        thisID = row['c_id']
       
        thisTheta = row['heading']
        # calculate the change in heading from this point to the next
        nextTime = posDF[(np.abs(posDF['frame']-(thisFrame+dt))<1e-6)&(posDF['c_id']==thisID)]
        if len(nextTime)==1:
            posDF.ix[index,'dtheta'] = nextTime.iloc[0]['heading'] -  thisTheta
            thisX = nextTime.iloc[0]['x']
            thisY = nextTime.iloc[0]['y']
            # calculate the average heading all the other caribou were taking at this point
            excThis = posDF[posDF.c_id!=thisID]
            xp = excThis['x'].values
            yp = excThis['y'].values
            xdirs = excThis['dx'].values
            ydirs = excThis['dy'].values
            kappa = 32.0*32.0
            dists = (((xp - thisX)**2 + (yp - thisY)**2))
            weights = np.exp(-dists/kappa)
            xav = np.sum(weights*xdirs)/np.sum(weights)
            yav = np.sum(weights*ydirs)/np.sum(weights)
            posDF.ix[index,'env_heading']  = math.atan2(yav,xav)-  thisTheta
            


    allDF = allDF.append(posDF,ignore_index=True)

    
allDF = allDF[np.isfinite(allDF['dtheta'])]
allDF = allDF.reset_index(drop=True)
dsize = len(allDF)
maxN=0
for index, row in allDF.iterrows():
    thisFrame =  row['frame']
    thisID = row['c_id']
    thisClip = row['clip']
    window = allDF[(allDF.frame==thisFrame)&(allDF['clip']==thisClip)&(allDF['c_id']!=thisID)]
    if len(window)>maxN:
        maxN=len(window)#

neighbours = np.zeros((dsize,maxN,3)).astype(np.float32) # dist, angle
#pixels are rescaled to meters based on flying at a height of 100m - camera fov = 60
px_to_m = 100*2.0*math.tan(math.radians(30))/1920.0

for index, row in allDF.iterrows():
    thisFrame =  row['frame']
    thisID = row['c_id']
    thisClip = row['clip']
    thisX = row['x']
    thisY = row['y']
    thisAngle = row['heading']
    window = allDF[(allDF.frame==thisFrame)&(allDF['clip']==thisClip)&(allDF['c_id']!=thisID)]
    ncount = 0

    for i2, w in window.iterrows():
        xj = w.x
        yj = w.y
        w_id = w['clip']*10000 + w['c_id']
        neighbours[index,ncount,0] = ((((thisX-xj)**2+(thisY-yj)**2))**0.5) * px_to_m 
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        neighbours[index,ncount,1] = math.atan2(math.sin(angle), math.cos(angle))
        neighbours[index,ncount,2] = w_id
        ncount+=1

# convert to a numpy array
allData = allDF.values


uid = allDF['clip'].values*10000 + allDF['c_id'].values 

mvector = allDF['dtheta'].values
mvector[mvector<-pi]=mvector[mvector<-pi]+2*pi
mvector[mvector>pi]=mvector[mvector>pi]-2*pi

evector = allDF['env_heading'].values
evector[evector<-pi]=evector[evector<-pi]+2*pi
evector[evector>pi]=evector[evector>pi]-2*pi

np.save('neighbours.npy', neighbours)
np.save('mvector.npy', mvector)
np.save('evector.npy', evector)
np.save('uid.npy', uid)
    

