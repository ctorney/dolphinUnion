import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'
LOGDIR = DATADIR + '/logs/'


# transitions for 2d movement with positions, velocity and acceleration
transition_matrix = [[1,0,1,0,0.5,0], [0,1,0,1,0,0.5], [0,0,1,0,1,0], [0,0,0,1,0,1], [0,0,0,0,1,0], [0,0,0,0,0,1]]

# observe only positions
observation_matrix = [[1,0,0,0,0,0], [0,1,0,0,0,0]]

# low noise on transitions
transition_covariance = np.eye(6)*1e-3
observation_covariance_m = np.eye(2)*2

kf = KalmanFilter(transition_matrices = transition_matrix, observation_matrices = observation_matrix, transition_covariance=transition_covariance,observation_covariance=observation_covariance_m)



df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/RELINKED_' + str(index) + '_' + noext + '.csv'
    outname = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    geoDF = pd.read_csv(geoName,index_col=0) 
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    
    fps = 60
    
    fStart = timeStart*fps
    

    # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame    
    alt = float(geoDF['alt'][fStart])/1000.0
    px_to_m = alt*2.0*math.tan(math.radians(30))/1920.0
    posDF = pd.read_csv(posfilename) 
    #break
    outTracks = pd.DataFrame(columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay','c_id'])
    
   
    for cnum, cpos2 in posDF.groupby('c_id'):
        cpos = cpos2.groupby(['frame','c_id'],as_index=False).agg(np.mean)
        obs = np.vstack((cpos['x'].values*px_to_m, cpos['y'].values*px_to_m)).T
        #obs = np.vstack((cpos['x'].values, cpos['y'].values)).T
        kf.initial_state_mean=[cpos['x'].values[0]*px_to_m,cpos['y'].values[0]*px_to_m,0,0,0,0]
        #kf.initial_state_mean=[cpos['x'].values[0],cpos['y'].values[0],0,0,0,0]

        sse = kf.smooth(obs)[0]

        ft = cpos['frame'].values
        xSmooth = sse[:,0]
        ySmooth = sse[:,1]
        xv = sse[:,2]/0.1
        yv = sse[:,3]/0.1
        xa = sse[:,4]/0.01
        ya = sse[:,5]/0.01
        dx = np.zeros_like(xSmooth)
        dy = np.zeros_like(xSmooth)
        headings = np.zeros_like(xSmooth)
        # calculate change in position for 5 second intervals
        for i in range(len(headings)):
            start = max(0,i-25)
            stop = min(i+25,len(headings))-1
            dx[i] = xSmooth[stop]-xSmooth[start]
            dy[i] = ySmooth[stop]-ySmooth[start]
        headings = np.arctan2(dy,dx)
        headings = np.arctan2(yv,xv)
        
        
        newcpos = pd.DataFrame(np.column_stack((ft,xSmooth,ySmooth,dx,dy,headings,xv,yv,xa,ya)), columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay'])
        newcpos['c_id']=cnum
        outTracks = outTracks.append(newcpos,ignore_index=True )
        
      

    #break
    outTracks.to_csv(outname, index=False)
    #break

