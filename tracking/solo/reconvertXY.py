
# repeat of conversion from pixel coordinates to real coordinates due to occasional glitch in original

import cv2
import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate


HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
MOVIEDIR = DATADIR + 'footage/' 
#MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'

df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    #if index==0:
    #    continue
    
    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/RELINKED_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    geoDF = pd.read_csv(geoName,index_col=0) 

    # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame    
    alt = float(geoDF['alt'][0])/1000.0
    mtopx = alt*2.0*math.tan(math.radians(30))/1920.0
    
    posDF = pd.read_csv(posfilename) 
    #posDF['x_px']=posDF['x']
    #posDF['y_px']=posDF['y']
    
    movieName = MOVIEDIR + row.filename
    warp_mode = cv2.MOTION_EUCLIDEAN
    cap = cv2.VideoCapture(movieName)
    
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps
    
    #fStart = 5508
    #fStop = 7308
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    number_of_iterations = 10;
             
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
    termination_eps = -1e-4;
                 
# Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    
    
    allTransforms = np.zeros((fStop-fStart,2,3))
    
    im1_gray = np.array([])
    
    warp_matrix = np.eye(2, 3, dtype=np.float32) 
    full_warp = np.eye(3, 3, dtype=np.float32)
    last_alt = float(geoDF['dtg'][fStart])#/1000.0
    for tt in range(fStart,fStop):

        _, frame = cap.read()
        
        if (tt%6)>0:
            continue
        alt = float(geoDF['dtg'][tt])#/1000.0
        if not(im1_gray.size):
 #           im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

        #im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        # find transform between this and previous frame
        (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    

        rescale=last_alt/alt
        last_alt=alt
        allTransforms[tt-fStart,:]=warp_matrix
        allTransforms[tt-fStart,0:2,0:2]=warp_matrix[0:2,0:2]*rescale
        #full_warp = np.dot(full_warp,np.vstack((warp_matrix,[0,0,1])))
        #im2_aligned = cv2.warpPerspective (im2_gray, full_warp, (S[0],S[1]), flags=cv2.INTER_LINEAR)
        #cv2.imwrite(str(tt) + '.png',im2_aligned)
        im1_gray =im2_gray.copy()
        
    full_warp = np.eye(3, 3, dtype=np.float32)
    for fnum, dfFrame in posDF.groupby('frame'):
        warp_matrix = allTransforms[fnum-fStart,:]
        # keep track of all the transformations to this point
        full_warp = np.dot(full_warp,np.vstack((warp_matrix,[0,0,1])))

        for ind,posrow in dfFrame.iterrows():
            # get pixel coordinates
            xx=posrow['x_px']
            yy=posrow['y_px']

            [mx,my,_] = np.dot(full_warp, np.array([[xx],[yy],[1]]))
            posDF.loc[ind,'x']=mx
            posDF.loc[ind,'y']=my
            #posDF.set_value(ind,'x',mx[0])
            #posDF.set_value(ind,'y',my)
            #print(mx,my)

        
    #c49=posDF[posDF['c_id']==49]      
    #plt.plot(c49.x[200:450],-c49.y[200:450])

    
    posDF.to_csv(posfilename)

