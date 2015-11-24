

import cv2
import numpy as np
import os,sys
import re
import math as math
import pandas as pd
import random


HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

show_index = 0
outputMovie=True

for index, row in df.iterrows():
    if index!=show_index:
        continue


    noext, ext = os.path.splitext(row.filename)   


    trackName = TRACKDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    posName = TRACKDIR + '/CARIBOU_POS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    movieName = MOVIEDIR + row.filename
    outputName = MOVIEDIR + '/TRACKS_' + str(index) + '_' + noext + '.avi'


    linkedDF = pd.read_csv(posName) 

    warp_mode = cv2.MOTION_EUCLIDEAN
    
    number_of_iterations = 100;
             
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
    termination_eps = 1e-3;
                 
# Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    
    
    
    cap = cv2.VideoCapture(movieName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = 4326# timeStart*fps
    fStop = 4329#4415# timeStop*fps
    
    for i in range(86):
        fStart = 4326 + i-40
        fStop = 4329 + i-40
        cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
        _, frame = cap.read()
        rows,cols,_ = frame.shape

        im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        cap.set(cv2.CAP_PROP_POS_FRAMES,fStop)
        _, frame = cap.read()
        im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)              
        (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)    
        print(math.degrees(math.acos(warp_matrix[0,0])))
#                                    
#    M = cv2.getRotationMatrix2D((cols/2,rows/2),360-277.53,1)
#    dst = cv2.warpAffine(frame,M,(cols,rows))
#    cv2.imwrite('im1f.png',frame)
#    cap.set(cv2.CAP_PROP_POS_FRAMES,fStop)
#    _, frame = cap.read()
#    cv2.imwrite('im2f.png',frame)
#    #rigid_mat = cv2.estimateRigidTransform(im1_gray,im2_gray, False)
#    #M = cv2.getRotationMatrix2D((cols/2,rows/2),360-298.02,1)
#    #M = cv2.getRotationMatrix2D((cols/2,rows/2),360-277.53,1)
#    
#    angleBetween2Coords = 276
#    xshift = -1.178#109.94*math.cos(math.radians(90-angleBetween2Coords))    
#    yshift = 12.2#109.94*math.sin(math.radians(90-angleBetween2Coords))    
#    M = np.float32([[1,0,-xshift],[0,1,-yshift]])
#    angle = -0.0054
#    angle = 0.0158680668
#
#    M = np.float32([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0]])
#    dst = cv2.warpAffine(frame,M,(cols,rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#    cv2.imwrite('shift.png',dst)
#
#    im2_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#                                 
#    (cc, warp_matrix) = cv2.findTransformECC (im1_gray[100:-100,100:-100],im2_gray[100:-100,100:-100],warp_matrix, warp_mode, criteria)    
#  
#   
##    dst2 = cv2.warpAffine(dst,M,(cols,rows))
#    im2_aligned = cv2.warpAffine(dst, warp_matrix, (1920,1080), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#                 
#    #gsout =  cv2.cvtColor(im2_aligned,cv2.COLOR_BGR2GRAY)
#    cv2.imwrite('rotated.png',im2_aligned)
#    
    cap.release()
    

