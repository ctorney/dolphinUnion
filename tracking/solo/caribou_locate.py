
import cv2
import numpy as np
import pandas as pd
import os,sys
import re
import time
import math


import pickle


from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



sys.path.append('./classify/.')

from circularHOGExtractor import circularHOGExtractor
ch = circularHOGExtractor(4,2,4) 

fhgClass = pickle.load( open( "./classify/svmClassifier2.p", "rb" ) )

def checkIsCaribou(x,y,frame,alt):
    nx = 1920
    ny = 1080
    # work out size of box if box if 32x32 at 100m
    grabSize = math.ceil((100.0/alt)*16.0)
    tmpImg =  cv2.cvtColor(frame[max(0,y-grabSize):min(ny,y+grabSize), max(0,x-grabSize):min(nx,x+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
    if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
        res = fhgClass.predict(ch.extract(cv2.resize(tmpImg,(32,32))))
        if res[0]>0.5:
            return True

    return False




HD = os.getenv('HOME')


DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
MOVIEDIR = HD + '/workspace/dolphinUnion/tracking/solo/'

params = cv2.SimpleBlobDetector_Params()

params.maxThreshold= 110 
params.minThreshold= 50
params.thresholdStep= 5
params.filterByArea= 1
params.maxArea= 500 
params.minArea= 3
params.filterByCircularity= 0
params.maxCircularity= 1
params.minCircularity= 0
params.filterByInertia= 0
params.filterByConvexity= 0

blobdetector = cv2.SimpleBlobDetector_create(params)

df = pd.read_csv(FILELIST)
outputMovie=0

for index, row in df.iterrows():
   
    if index!=3:
        continue
    # pandas file for export of positions
    dfPos = pd.DataFrame(columns= ['x', 'y', 'frame'])    
    
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    inputName = MOVIEDIR + row.filename
    

    noext, ext = os.path.splitext(row.filename)
    posfilename = DATADIR + 'tracked/CARIBOU_POS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    geoDF = pd.read_csv(geoName) 
    
    cap = cv2.VideoCapture(inputName)
    S = (1920,1080)
    if outputMovie:
        out = cv2.VideoWriter('tmp.avi', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS)/6, S, True)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    
    for tt in range(fStart, fStop):
        
        print(tt,fStop)
        # Capture frame-by-frame
        _, frame = cap.read()
        if frame is None:
            break
        # movies are 60 fps so cut down to 10 fps
        if (tt%6)!=0:
            continue
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs= blobdetector.detect(cv2image)
        # draw detected objects and display
        sz=6
        thisFrame = pd.DataFrame(columns= ['x', 'y', 'frame'])
        # get altitude of frame
        alt = float(geoDF['alt'][tt])/1000.0            
        ind = 0
        for b in blobs:
            if checkIsCaribou(b.pt[0],b.pt[1],frame,alt):
                if outputMovie:
                    cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
                thisFrame.set_value(ind, 'x', b.pt[0])
                thisFrame.set_value(ind, 'y', b.pt[1])
                thisFrame.set_value(ind, 'frame', tt)
                ind +=1
        dfPos = pd.concat([dfPos,thisFrame])
        
        ## output
        if outputMovie:
            out.write(frame)
        

    cap.release()
    if outputMovie:
        out.release()

    dfPos.to_csv(posfilename)

