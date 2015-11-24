


import numpy as np
import pandas as pd
import os
import math
import time
from scipy import interpolate
import LatLon

def convertPixelsToMeters(px,py,m_to_px,hdg,camx,camy):
     # centre point of camera    
    cx = 960 
    cy = 540
    
    # distance of caibou to centre of frame in metres
    xmetres = (px-cx)
    ymetres = (py-cy)
    
    # rotate due to heading of UAV    
    xreal =cx + xmetres*math.cos(hdg) -  ymetres*math.sin(hdg)
    yreal = 1080- (cy + ymetres*math.cos(hdg) +  xmetres*math.sin(hdg))


    #return [camx, camy]
    return [camx+(m_to_px)*xreal,camy+(m_to_px)*yreal]


HD = os.getenv('HOME')

DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():
    if index!=0:
        continue

    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/CARIBOU_POS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    geoDF = pd.read_csv(geoName,index_col=0) 
    posDF = pd.read_csv(posfilename) 
    posDF = posDF.drop('Unnamed: 0', 1)
    posDF['x_px']=posDF['x']
    posDF['y_px']=posDF['y']
    
    # use start of frame as the origin
    startLat = geoDF['lat'][posDF['frame'][0]]/1.0e7
    startLon = geoDF['lon'][posDF['frame'][0]]/1.0e7
    startPoint = LatLon.LatLon(LatLon.Latitude(startLat), LatLon.Longitude(startLon))
    
    for fnum, frame in posDF.groupby('frame'):
        fn = fnum #+ 12
        alt = float(geoDF['alt'][fn])/1000.0
        lat = float(geoDF['lat'][fn])/1.0e7
        lon = float(geoDF['lon'][fn])/1.0e7
        hdg = -math.radians(360-float(geoDF['hdg'][fn])/100.0)
        thisPoint = LatLon.LatLon(LatLon.Latitude(lat), LatLon.Longitude(lon))
        dist = startPoint.distance(thisPoint)*1000 # distance in metres
        theta = startPoint.heading_initial(thisPoint)
        camx = dist*math.cos(math.radians(90-theta))
        camy = dist*math.sin(math.radians(90-theta))
        #if (fnum-3120)/60>50 and (fnum-3120)/60<74:
        #    print((fnum-3120)/60,dist)
            
        
        # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame
        mtopx = alt*2.0*math.tan(math.radians(30))/1920.0
        
        first = 1
        for ind,row in frame.iterrows():
            xx0=row['x_px']
            yy0=row['y_px']

            [mx,my] = convertPixelsToMeters(row['x_px'],row['y_px'],mtopx,hdg, camx, camy)
            #if fnum==3996:
            #    print(xx0,yy0,mx,my)
            if first:
                mx=-camx
                my=-camy
                first=0
                
            posDF.set_value(ind,'x',mx)
            posDF.set_value(ind,'y',my)

                


    
    posDF.to_csv('test.csv')

