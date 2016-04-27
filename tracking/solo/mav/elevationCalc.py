import googlemaps
import numpy as np
import pandas as pd
import os, re
import math
import time
import csv
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'
LOGDIR = DATADIR + '/logs/'

gmaps = googlemaps.Client(key='AIzaSyCCbySu2DPDdByQCNGjWg_BZdHj6e5Favc')

df = pd.read_csv(FILELIST)
for index, row in df.groupby('filename'):
    noext, ext = os.path.splitext(row.filename.iloc[0])   
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    tlogName = LOGDIR + '/LOG_' + noext + '.csv'
    
    
    # open the file read in all the data and record the timestamp, recording start, recording stop
    csvfile = open(tlogName, 'r') 
    rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for logrow in rowreader:
            
        if logrow[1]=='GLOBAL_POSITION_INT':
            # POSITION, HEIGHT, HEADING ETC from the autopilot - values are corrected 
            if int(logrow[11])>500:
                startLat = int(logrow[5])
                startLon = int(logrow[7])

    geoDF = pd.read_csv(geoName,index_col=0) 
    gmapData = gmaps.elevation((startLat/10000000.0,startLon/10000000))
    startE = gmapData[0]['elevation']

    lats = geoDF[geoDF['time']%1==0]['lat'].values/10000000.0
    lons = geoDF[geoDF['time']%1==0]['lon'].values/10000000.0
    times = geoDF[geoDF['time']%1==0]['time'].values
    ESL = np.array([])
    jsize=300
    for g_index in range(0,len(times),jsize): 
        # get elevation above sea level at each location
        gmapData = gmaps.elevation(list(zip(lats[g_index:g_index+jsize],lons[g_index:g_index+jsize])))
        # convert to an array
        ESL = np.hstack((ESL,[points['elevation'] for points in gmapData]))
    # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame    
    #plt.plot(ESL)
    #plt.show()
    
    interESL = interpolate.interp1d(np.append(times,times[-1]+1), np.append(ESL,ESL[-1]))(geoDF['time'].values)
    geoDF['dtg'] = 0
    for g_index, g_row in geoDF.iterrows():
        geoDF.loc[g_index,'dtg']=(geoDF.loc[g_index,'alt']/1000.0+startE-interESL[g_index])
         
    geoDF.to_csv(geoName)




