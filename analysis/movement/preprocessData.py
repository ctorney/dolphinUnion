
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
for indexF, rowF in df.iterrows():
    noext, ext = os.path.splitext(rowF.filename)   
    posfilename = OUTDIR + '/TRACKS_' + str(indexF) + '_' + noext + '.csv'
    gridfilename = DATADIR + '/GRID_' + str(indexF) + '_' + noext + '.npy'
    gridPosfilename = DATADIR + '/GRIDPOS_' + str(indexF) + '_' + noext + '.npy'
    posDF = pd.read_csv(posfilename) 
    posDF['clip']=indexF
    posDF = posDF[posDF['frame']%120==0]
    dt = 60 # 60 frames is 1 second
    posDF['move']=np.NaN
    posDF['moveLength']=np.NaN
    posDF['env_heading']=np.NaN
    for index, row in posDF.iterrows():
        thisFrame =  row['frame']
        thisID = row['c_id']
        thisX = row['x']
        thisY = row['y']
            
        thisTheta = row['heading']
        # calculate the change in heading from this point to the next
        nextTime = posDF[(np.abs(posDF['frame']-(thisFrame+2*dt))<1e-6)&(posDF['c_id']==thisID)]
        if len(nextTime)==1:
            # calculate the average heading all the other caribou were taking at this point
            excThis = posDF[posDF.c_id!=thisID]
            xp = excThis['x'].values
            yp = excThis['y'].values
            xdirs = np.cos(excThis['heading'].values)
            ydirs = np.sin(excThis['heading'].values)
            # decay rate of 3 gives the maximum likelihood for the environment only model
            kappa = 3.0**2
            dists = (((xp - thisX)**2 + (yp - thisY)**2))
            weights = np.exp(-dists/kappa)
            if np.sum(weights)>0:
                xav = np.sum(weights*xdirs)/np.sum(weights)
                yav = np.sum(weights*ydirs)/np.sum(weights)
                posDF.ix[index,'env_heading']  = math.atan2(yav,xav)-  thisTheta
            else:
                posDF.ix[index,'env_heading']  = 0.0
            
            # 1 second move heading
            dx = nextTime.iloc[0]['x'] - thisX
            dy = nextTime.iloc[0]['y'] - thisY
            posDF.ix[index,'dx']=dx
            posDF.ix[index,'dy']=dy
            posDF.ix[index,'move'] = math.atan2(dy,dx) -  thisTheta
            posDF.ix[index,'moveLength'] = (dy**2+dx**2)**0.5

            


    allDF = allDF.append(posDF,ignore_index=True)



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

neighbours = np.zeros((dsize,maxN,5)).astype(np.float32) # dist, angle
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
        dx = xj - thisX
        dy = yj - thisY
        
        neighbours[index,ncount,0] = ((((dx)**2+(dy)**2))**0.5) #* px_to_m 
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle

        neighbours[index,ncount,1] = math.atan2(math.sin(angle), math.cos(angle))
        w_id = w['clip']*10000 + w['c_id']
        neighbours[index,ncount,2] = w_id
        jdx = w.dx
        jdy = w.dy
        jAngle = math.atan2(jdy,jdx) - thisAngle
        neighbours[index,ncount,3] = jAngle
        jMoveLength = ((jdx**2)+(jdy**2))**0.5
        neighbours[index,ncount,4] = jMoveLength
        
        ncount+=1

# convert to a numpy array
#allData = allDF.values
#keep non nan moves, of more than 1m and less than 10m
keepIndexes = (np.isfinite(allDF['move'].values))&(allDF['moveLength'].values>1)&(allDF['moveLength'].values<10)

uid = allDF['clip'].values*10000 + allDF['c_id'].values 

np.save('pdata/neighbours.npy', neighbours[keepIndexes])
np.save('pdata/uid.npy', uid[keepIndexes])

mvector = allDF['move'].values
mvector[mvector<-pi]=mvector[mvector<-pi]+2*pi
mvector[mvector>pi]=mvector[mvector>pi]-2*pi
np.save('pdata/mvector.npy', mvector[keepIndexes])



evector = allDF['env_heading'].values
evector[evector<-pi]=evector[evector<-pi]+2*pi
evector[evector>pi]=evector[evector>pi]-2*pi
np.save('pdata/evector.npy', evector[keepIndexes])
    


