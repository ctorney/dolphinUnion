

import cv2
import numpy as np
import os,sys
import re
import math as m
import pandas as pd
import random


HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

show_index = 5
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


    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    linkedDF = pd.read_csv(posName) 

    

    
    
    
    cap = cv2.VideoCapture(movieName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    
    if outputMovie:
        out = cv2.VideoWriter('tmp'+str(random.randint(0,10000))+ '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS)/6, S, True)
    
    for tt in range(fStart,fStop):

        _, frame = cap.read()
        if (tt%6) > 0 : continue
            
        thisFrame = linkedDF.ix[linkedDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=6
        
        for i, trrow in thisFrame.iterrows():
            print(i)
           # cv2.putText(frame ,str(int(trrow['particle'])) ,((int(trrow['x'])+12, int(trrow['y'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
            cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
        
        if outputMovie:
            out.write(frame)
            
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()

