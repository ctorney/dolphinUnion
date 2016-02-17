import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
OUTDIR =  HD + '/Dropbox/dolphin_union/2015_footage/Solo/processedTracks/'

df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/FINAL_' + str(index) + '_' + noext + '.csv'
    outname = OUTDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'

    posDF = pd.read_csv(posfilename) 
    
    outTracks = pd.DataFrame(columns= ['frame','x','y','heading','vx','vy','ax','ay','c_id'])
    
    #smoothing
    winLen = 10
    vwinLen = 30
    w = np.kaiser(winLen,1)
    w = w/w.sum()
    w2 = np.kaiser(vwinLen,1)
    w2 = w2/w2.sum()
    for cnum, cpos in posDF.groupby('c_id'):
        ft = cpos['frame'].values
        xd = cpos['x'].values
        xd = np.r_[np.ones((winLen))*xd[0],xd,np.ones((winLen))*xd[-1]]
        xSmooth = np.convolve(w/w.sum(),xd,mode='same')[(winLen):-(winLen)]
        xv = np.diff(xSmooth)
        xv = np.r_[np.ones((vwinLen))*xv[0],xv,np.ones((vwinLen))*xv[-1]]
        xv = np.convolve(w2/w2.sum(),xv,mode='same')[(vwinLen):-(vwinLen-1)]
        xa = np.diff(xv)
        xa = np.r_[np.ones((vwinLen))*xa[0],xa,np.ones((vwinLen))*xa[-1]]
        xa = np.convolve(w2/w2.sum(),xa,mode='same')[(vwinLen):-(vwinLen-1)]
        yd = cpos['y'].values
        yd = np.r_[np.ones((winLen))*yd[0],yd,np.ones((winLen))*yd[-1]]
        ySmooth = np.convolve(w/w.sum(),yd,mode='same')[(winLen):-(winLen)]
        yv = np.diff(ySmooth)
        yv = np.r_[np.ones((vwinLen))*yv[0],yv,np.ones((vwinLen))*yv[-1]]
        yv = np.convolve(w2/w2.sum(),yv,mode='same')[(vwinLen):-(vwinLen-1)]
        ya = np.diff(yv)
        ya = np.r_[np.ones((vwinLen))*ya[0],ya,np.ones((vwinLen))*ya[-1]]
        ya = np.convolve(w2/w2.sum(),ya,mode='same')[(vwinLen):-(vwinLen-1)]
        #xSmooth = xSmooth[(winLen):-(winLen)]
        headings = np.zeros_like(xSmooth)
        dx = xSmooth[1:]-xSmooth[0:-1]
        dy = ySmooth[1:]-ySmooth[0:-1]
        headings[0:-1] = np.arctan2(dy,dx)
        headings[-1]=headings[-2] 
#        plot arrows for error checking
#        x=xSmooth[0:-1]
#        y=ySmooth[0:-1]
#        u = np.cos(headings)
#        v = np.sin(headings)
#        plt.quiver(x, y, u ,v) 
#        plt.axes().set_aspect('equal')
#        plt.show()

        newcpos = pd.DataFrame(np.column_stack((ft,xSmooth,ySmooth,headings,xv,yv,xa,ya)), columns= ['frame','x','y','heading','vx','vy','ax','ay'])
        newcpos['c_id']=cnum
        outTracks = outTracks.append(newcpos,ignore_index=True )
        
       # for ind,posrow in dfFrame.iterrows():
       #     # get pixel coordinates
       #     xx=posrow['x_px']
       #     yy=posrow['y_px']
            
       #     [mx,my,_] = np.dot(full_warp, np.array([[xx],[yy],[1]]))
       #     posDF.set_value(ind,'x',mx)
       #     posDF.set_value(ind,'y',my)
        
          


    
    outTracks.to_csv(outname, index=False)

