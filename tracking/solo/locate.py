
import cv2
import numpy as np
import pandas as pd
import os
import re
import time

HD = os.getenv('HOME')


#DATADIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

params = cv2.SimpleBlobDetector_Params()
params.maxThreshold= 120
params.minThreshold= 10
params.thresholdStep= 5
params.minDistBetweenBlobs= 1
params.filterByArea= 1
params.maxArea= 500
params.minArea= 15
params.filterByCircularity= 0
params.filterByInertia= 0
params.filterByConvexity= 0

blobdetector = cv2.SimpleBlobDetector_create(params)

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():
   
    # pandas file for export of positions
    dfPos = pd.DataFrame(columns= ['x', 'y', 'frame'])    
    
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    inputName = DATADIR + 'footage/' + row.filename
    

    noext, ext = os.path.splitext(row.filename)
    tlogName = DATADIR + 'logs/LOG_' + noext + '.csv'
    posfilename = DATADIR + 'tracked/TRACKS_' + noext + '.csv'
    
    print('Movie ' +  tlogName + ' from ' + str(timeStart) + ' to ' + str(timeStop))
    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    for tt in range(fStop-fStart):
        
        print(tt,fStop-fStart)
        # Capture frame-by-frame
        _, frame = cap.read()
        if frame is None:
            break
        # movies are 60 fps so cut down to 4 fps
        if (tt%15)!=0:
            continue
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs= blobdetector.detect(cv2image)
        # draw detected objects and display
        sz=6
        thisFrame = pd.DataFrame(columns= ['x', 'y', 'frame'])
        ind = 0
        for b in blobs:
            ind +=1
#            cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
            thisFrame.set_value(ind, 'x', b.pt[0])
            thisFrame.set_value(ind, 'y', b.pt[1])
            thisFrame.set_value(ind, 'frame', tt+fStart)
        dfPos = pd.concat([dfPos,thisFrame])
        
        ## output
        #out.write(frame)
        

    cap.release()
    #out.release()

    dfPos.to_csv(posfilename)

