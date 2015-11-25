

import cv2
import numpy as np
import pandas as pd
import os, re
import math
import time
from scipy import interpolate


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
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():
    if index!=0:
        continue

    noext, ext = os.path.splitext(row.filename)   
    posfilename = DATADIR + 'tracked/CARIBOU_POS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    geoDF = pd.read_csv(geoName,index_col=0) 

    # calculate the pixels to metre conversion by using the altitude, fov of camera, and width of frame    
    alt = float(geoDF['alt'][0])/1000.0
    mtopx = alt*2.0*math.tan(math.radians(30))/1920.0
    
    posDF = pd.read_csv(posfilename) 
    posDF = posDF.drop('Unnamed: 0', 1)
    posDF['x_px']=posDF['x']
    posDF['y_px']=posDF['y']
    
    
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
    #fStart = timeStart*fps
    #fStop = timeStop*fps
    fStop = fStart+7200
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    number_of_iterations = 10;
             
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
    termination_eps = -1e-3;
                 
# Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    
    
    allTransforms = np.zeros((fStop-fStart,3))
    #fStart=4380
    #cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    im1_gray = np.array([])
    lf = max(posDF['frame'])
    warp_matrix = np.eye(2, 3, dtype=np.float32) 
    for tt in range(fStart,fStop):

        _, frame = cap.read()
        
        if (tt%6)>0:
            continue
        
        if not(im1_gray.size):
            im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        
        im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
                     
        #(cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
        warp_matrix = cv2.estimateRigidTransform(im2_gray, im1_gray, False)
        print(tt,fStop)#,cc)
        allTransforms[tt-fStart,0]=math.atan2(warp_matrix[1,0],warp_matrix[0,0])
        allTransforms[tt-fStart,1]=warp_matrix[0,2]
        allTransforms[tt-fStart,2]=warp_matrix[1,2]
        #full_warp = np.dot(np.vstack((warp_matrix,[0,0,1])),full_warp)

        im2_aligned = cv2.warpAffine(frame, warp_matrix[0:2,:], (1920,1080), flags=cv2.INTER_LINEAR)# + cv2.WARP_INVERSE_MAP)   
        cv2.imwrite('frames/f-' + str(tt) + '.png',cv2.addWeighted(im2_aligned,0.7, cv2.cvtColor(im1_gray,cv2.COLOR_GRAY2BGR) ,0.3,0) )
        
        im1_gray =im2_gray.copy()
        
        #for ind,posrow in dfFrame.iterrows():
        #    xx0=posrow['x_px']
        #    yy0=posrow['y_px']

            #[mx,my,_] = np.dot(full_warp, np.array([[posrow['x_px']],[posrow['y_px']],[1]]))
            #if fnum==3996:
            #    print(xx0,yy0,mx,my)
            #if first:
            #    mx=-camx
            #    my=-camy
            #    first=0
                
         #   posDF.set_value(ind,'x',mx)
         #   posDF.set_value(ind,'y',my)
    
    #break
    full_warp = np.eye(3, 3, dtype=np.float32)
    for fnum, dfFrame in posDF.groupby('frame'):
#        if fnum%10>0:
#            continue
#        if fnum<3660:
#            continue
#        print(fnum,lf)
#        cap.set(cv2.CAP_PROP_POS_FRAMES,fnum)
#        _, frame = cap.read()
#        if not(im1_gray.size):
#            im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#
#        
#        im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#        
#        warp_matrix = np.eye(2, 3, dtype=np.float32)              
#        (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
#        
#        full_warp = np.dot(np.vstack((warp_matrix,[0,0,1])),full_warp)
#
#        im2_aligned = cv2.warpAffine(frame, full_warp[0:2,:], (1920,1080), flags=cv2.INTER_LINEAR)# + cv2.WARP_INVERSE_MAP)   
#        cv2.imwrite('frames/f-' + str(fnum) + '.png',im2_aligned)
#        
#        im1_gray =im2_gray.copy()
        
        for ind,posrow in dfFrame.iterrows():
            xx0=posrow['x_px']
            yy0=posrow['y_px']
            dth = np.sum(allTransforms[0:fnum-fStart,0])
            dx = np.sum(allTransforms[0:fnum-fStart,1])
            dy = np.sum(allTransforms[0:fnum-fStart,2])
            
            full_warp[0,0]=math.cos(dth)
            full_warp[0,1]=math.sin(dth)
            full_warp[1,0]=-math.sin(dth)
            full_warp[1,1]=math.cos(dth)
            full_warp[0,2]=-dx
            full_warp[1,2]=-dy
            
            cx = 960 
            cy = 540
    
            # distance of caibou to centre of frame in metres
            xpre = (xx0-cx)
            ypre = (yy0-cy)
    
    # rotate due to heading of UAV    
    #xreal =cx + xmetres*math.cos(hdg) -  ymetres*math.sin(hdg)
    #yreal = 1080- (cy + ymetres*math.cos(hdg) +  xmetres*math.sin(hdg))
            [mx,my,_] = np.dot(full_warp, np.array([[xpre],[ypre],[1]]))
            #if fnum==3996:
            #    print(xx0,yy0,mx,my)
            #if first:
            #    mx=-camx
            #    my=-camy
            #    first=0
                
            posDF.set_value(ind,'x',cx+mx)
            posDF.set_value(ind,'y',cy+my)
        
          


    
    posDF.to_csv('test.csv')

