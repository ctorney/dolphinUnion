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
    print(geoName)
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

    break
#    ESL = np.empty(len(geoDF))
#    for g_index, g_row in geoDF.iterrows(): 
#        print(g_index)
#        # get elevation above sea level at each location
#        #gmapData = gmaps.elevation(list(zip(geoDF['lat'][0:200].values/10000000.0,(geoDF['lon'][0:200].values)/10000000)))
#        gmapData = gmaps.elevation((g_row['lat']/10000000.0,g_row['lon']/10000000))
#        ESL[g_index] = gmapData[0]['elevation']
#        # convert to an array
#        #ESL = [points['elevation'] for points in gmapData]
    ESL = np.array([])
    jsize=330
    for g_index in range(0,len(geoDF),jsize): 
        print(g_index,len(geoDF))
        # get elevation above sea level at each location
        gmapData = gmaps.elevation(list(zip(geoDF['lat'][g_index:g_index+jsize].values/10000000.0,(geoDF['lon'][g_index:g_index+jsize].values)/10000000)))
        # convert to an array
        ESL = np.hstack((ESL,[points['elevation'] for points in gmapData]))
    # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame    
    plt.plot(ESL)
    plt.show()
    





