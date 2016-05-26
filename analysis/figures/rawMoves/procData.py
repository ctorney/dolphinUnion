import os
import csv
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import pandas as pd
import math
from matplotlib import transforms

 
HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

df = pd.read_csv(FILELIST)
allDF = pd.DataFrame()
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'

    posDF = pd.read_csv(posfilename) 
    posDF['clip']=index
    posDF = posDF[posDF['frame']%60==0]

    allDF = allDF.append(posDF,ignore_index=True)
    

        
# convert to a numpy array
allData = allDF.values


rowCount = len(allData)


# build an array to store the relative angles and distances to all neighbours
locations = np.zeros((0,5)).astype(np.float32) 
for thisRow in range(rowCount):
    thisTime = allData[thisRow,0]        
    thisX = allData[thisRow,1]
    thisY = allData[thisRow,2]
    thisAngle = (allData[thisRow,5])
    thisTrack = (allData[thisRow,10])
    thisClip = allData[thisRow,11]
    # find all animals at this time point in the clip that aren't the focal individual
    window = allData[(allData[:,0]==thisTime)&(allData[:,11]==thisClip)&(allData[:,10]!=thisTrack),:]
    rowLoc = np.zeros((0,5)).astype(np.float32) 
    for w in window:
        xj = w[1]
        yj = w[2]
        jAngle = (w[5])
        r = ((((thisX-xj)**2+(thisY-yj)**2))**0.5)
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        jAngle = jAngle - thisAngle
        axj=w[8] # accelaration in x-direction
        ayj=w[9] # accelaration in y-direction
        accangle = math.atan2(ayj,axj)
        acclength = (ayj**2+axj**2)**0.5
        accangle = accangle - thisAngle
        #angle = math.atan2(dy,dx)
        theta = math.atan2(math.sin(angle), math.cos(angle))
        jHeading  = math.atan2(math.sin(jAngle), math.cos(jAngle))
        rowLoc = np.vstack((rowLoc,[r, theta, jHeading, accangle,acclength]))
    locations = np.vstack((locations,rowLoc))


np.save('locations.npy',locations)
