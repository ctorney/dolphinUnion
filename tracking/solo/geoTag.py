

import csv
import numpy as np
import pandas as pd
import os
import re
import time
import datetime
from scipy import interpolate
HD = os.getenv('HOME')


#DATADIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():
    if index!=4:
        continue

    noext, ext = os.path.splitext(row.filename)   
    
    # ********* read in log file and extract position data ********* 

    tlogName = DATADIR + '/LOG_' + noext + '.csv'
    
    # arrays for storing data/timestamps
    tData = np.zeros(shape=(0,6))
    timestamp = []
    
    # open the file read in all the data and record the timestamp, recording start, recording stop
    csvfile = open(tlogName, 'r') 
    lastHB = []
    rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for logrow in rowreader:
        if logrow[1]=='GOPRO_HEARTBEAT':
            # HEARTBEAT indicates if the gopro is recording (value=3) or not (value=2)
            # so we look for times when it has switched from 2 to 3 and vice versa
            if not(len(lastHB)):
                lastHB = logrow
            if int(logrow[3])==3 and int(lastHB[3])==2:
                #started recording sometime in the previous interval so will take the midpoint between the two timestamps as the start time
                recStart = datetime.datetime.strptime(str(lastHB[0]), '%d-%m-%Y:%H:%M:%S.%f') +  0.5*(datetime.datetime.strptime(str(logrow[0]), '%d-%m-%Y:%H:%M:%S.%f') - datetime.datetime.strptime(str(lastHB[0]), '%d-%m-%Y:%H:%M:%S.%f'))
            if int(logrow[3])==2 and int(lastHB[3])==3:
                recStop =  datetime.datetime.strptime(str(lastHB[0]), '%d-%m-%Y:%H:%M:%S.%f') + 0.5*(datetime.datetime.strptime(str(logrow[0]), '%d-%m-%Y:%H:%M:%S.%f') - datetime.datetime.strptime(str(lastHB[0]), '%d-%m-%Y:%H:%M:%S.%f'))
                #stopped recording
            lastHB = logrow
            
        elif logrow[1]=='GLOBAL_POSITION_INT':
            # POSITION, HEIGHT, HEADING ETC from the autopilot - values are corrected 
            rowVals = np.array([int(logrow[5]),int(logrow[7]),int(logrow[11]),int(logrow[13]),int(logrow[15]),int(logrow[19])])
            tData = np.vstack((tData,rowVals))
            timestamp.append(str(logrow[0]))

    # build an array with time since record start 
    recTime = np.zeros_like(tData[:,0]).astype(np.float64)-1.0 # time since recording starting in seconds
    for index in range(len(timestamp)):
        #if (datetime.datetime.strptime(timestamp[index], '%d-%m-%Y:%H:%M:%S.%f') < recStop):
        recTime[index] = (datetime.datetime.strptime(timestamp[index], '%d-%m-%Y:%H:%M:%S.%f') - recStart).total_seconds()

    # keep values where we're recording
    recordingIndexes = recTime>=0

    # resample to the log values align with the times of frames
    videoLength = np.max(recTime)
    
    frameTimes = np.arange(0,videoLength,1.0/60)    
    
    tData = interpolate.interp1d(recTime, tData.T)(frameTimes)
    tData=tData.T

    # frame list with positional and altitude data
    geofilename = DATADIR + '/GEOTAG_' + noext + '.csv'

    # pandas file for export of geotagging data
    dfGeo = pd.DataFrame(np.column_stack((frameTimes,tData)), columns= ['time','lat', 'lon', 'alt','vx','vy','hdg'])  
    dfGeo.to_csv(geofilename)

