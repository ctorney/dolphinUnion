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



# transitions for 2d movement with positions, velocity and acceleration
transition_matrix = [[1,0,1,0,0.5,0], [0,1,0,1,0,0.5], [0,0,1,0,1,0], [0,0,0,1,0,1], [0,0,0,0,1,0], [0,0,0,0,0,1]]

# observe only positions
observation_matrix = [[1,0,0,0,0,0], [0,1,0,0,0,0]]

# low noise on transitions
transition_covariance = np.eye(6)*1e-3
observation_covariance_m = np.eye(2)*2

kf = KalmanFilter(transition_matrices = transition_matrix, observation_matrices = observation_matrix, transition_covariance=transition_covariance,observation_covariance=observation_covariance_m)
#plt.plot(a[:,0],-a[:,1])
#
#plt.plot(smoothed_state_estimates[:,0],-smoothed_state_estimates[:,1])
#plt.plot(smoothed_state_estimates[:,2])
#plt.plot(smoothed_state_estimates[:,4])
#plt.figure()
#bb = np.arctan2(smoothed_state_estimates[:,3],smoothed_state_estimates[:,2])
# draw estimates
#pl.figure()
#lines_true = pl.plot(states, color='b')
#lines_filt = pl.plot(filtered_state_estimates, color='r')
#lines_smooth = pl.plot(smoothed_state_estimates, color='g')
#pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
#          ('true', 'filt', 'smooth'),
#          loc='lower right'
#)
#pl.show()
px_to_m = 100*2.0*math.tan(math.radians(30))/1920.0

df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/RELINKED_' + str(index) + '_' + noext + '.csv'
    outname = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'

    posDF = pd.read_csv(posfilename) 
    #break
    outTracks = pd.DataFrame(columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay','c_id'])
    
   
    for cnum, cpos2 in posDF.groupby('c_id'):
        cpos = cpos2.groupby(['frame','c_id'],as_index=False).agg(np.mean)
        obs = np.vstack((cpos['x'].values, cpos['y'].values)).T
        
        kf.initial_state_mean=[cpos['x'].values[0],cpos['y'].values[0],0,0,0,0]

        sse = kf.smooth(obs)[0]

        ft = cpos['frame'].values
        xSmooth = sse[:,0]
        ySmooth = sse[:,1]
        xv = sse[:,2]*px_to_m/0.1
        yv = sse[:,3]*px_to_m/0.1
        xa = sse[:,4]*px_to_m/0.01
        ya = sse[:,5]*px_to_m/0.01
        headings = np.arctan2(yv,xv)
        dx = np.zeros_like(xSmooth)
        dy = np.zeros_like(xSmooth)
        # calculate change in position for 5 second intervals
        for i in range(len(headings)):
            start = max(0,i-25)
            stop = min(i+25,len(headings))-1
            dx[i] = xSmooth[stop]-xSmooth[start]
            dy[i] = ySmooth[stop]-ySmooth[start]
        
        #headings[-1]=headings[-2] 
#        plot arrows for error checking
#        x=xSmooth[0:-1]
#        y=ySmooth[0:-1]
#        u = np.cos(headings)
#        v = np.sin(headings)
#        plt.quiver(x, y, u ,v) 
#        plt.axes().set_aspect('equal')
#        plt.show()

        newcpos = pd.DataFrame(np.column_stack((ft,xSmooth,ySmooth,dx,dy,headings,xv,yv,xa,ya)), columns= ['frame','x','y','dx','dy','heading','vx','vy','ax','ay'])
        newcpos['c_id']=cnum
        outTracks = outTracks.append(newcpos,ignore_index=True )
        
       # for ind,posrow in dfFrame.iterrows():
       #     # get pixel coordinates
       #     xx=posrow['x_px']
       #     yy=posrow['y_px']
            
       #     [mx,my,_] = np.dot(full_warp, np.array([[xx],[yy],[1]]))
       #     posDF.set_value(ind,'x',mx)
       #     posDF.set_value(ind,'y',my)
        
    #    break


    #break
    outTracks.to_csv(outname, index=False)
    break

