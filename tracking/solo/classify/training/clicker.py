

import cv2
import numpy as np
import os,sys
import math 
import pandas as pd
import re
import random

def is_caribou(event,x,y,flags,param):
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_path = "yes2/img-" + noext + '_' + str(counter) + ".png"
        # get altitude of frame
        # work out size of box if box if 32x32 at 100m
        grabSize = math.ceil((100.0/alt)*16.0)
        tmpImg =  cv2.cvtColor(frame[max(0,y-grabSize):min(ny,y+grabSize), max(0,x-grabSize):min(nx,x+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
        if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
            cv2.imwrite(save_path,cv2.resize(tmpImg,(32,32)))
            #cv2.imwrite('./yes/' + noext + '_' + str(caribou[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
            
        counter += 1
        
def isnt_caribou(event,x,y,flags,param):
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_path = "no2/img-" + noext + '_' + str(counter) + ".png"
        # get altitude of frame
        # work out size of box if box if 32x32 at 100m
        grabSize = math.ceil((100.0/alt)*16.0)
        tmpImg =  cv2.cvtColor(frame[max(0,y-grabSize):min(ny,y+grabSize), max(0,x-grabSize):min(nx,x+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
        if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
            cv2.imwrite(save_path,cv2.resize(tmpImg,(32,32)))
            #cv2.imwrite('./yes/' + noext + '_' + str(caribou[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
            
        counter += 1










nx = 1920
ny = 1080

counter = random.randint(0,10000)

HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)
endClass = False
for index, row in df.iterrows():

    if index!=2:
        continue
    
    noext, ext = os.path.splitext(row.filename)   


    trackName = TRACKDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    movieName = MOVIEDIR + row.filename

    geoDF = pd.read_csv(geoName) 
    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    

    cap = cv2.VideoCapture(movieName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    for tt in range(fStart, fStop):
    
    
        alt = float(geoDF['alt'][tt])/1000.0            

        _, frame = cap.read()
        if frame is None:
            break
        
        if (tt%60)!=0:
            continue
        box_dim = 128    
    
        sz=16
    
        cv2.destroyAllWindows()
        frName = 'dbl click caribou (ESC to quit, c to continue)'
        cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(frName,is_caribou)
        cv2.imshow(frName,frame)
        while(1):
            k = cv2.waitKey(0)
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        cv2.destroyAllWindows()
        if endClass: break
        frName = 'dbl click non-caribou (ESC to quit, c to continue)'
        cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(frName,isnt_caribou)
        cv2.imshow(frName,frame)
        while(1):
            k = cv2.waitKey(0)
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        cv2.destroyAllWindows()
        if endClass: break
                

    
            
            
    
    
    
    cap.release()



