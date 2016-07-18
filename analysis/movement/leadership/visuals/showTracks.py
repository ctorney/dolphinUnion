

import cv2
import numpy as np
import os,sys
import re
import math 
import pandas as pd
import random
def cvDrawDottedLine(img, pt1, pt2, color, thickness, lengthDash, lengthGap):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    
    start = 0
    while start<dist:
        stop=min(dist,start+lengthDash)
        x1=int((pt1[0]*(1-start/dist)+pt2[0]*start/dist))
        y1=int((pt1[1]*(1-start/dist)+pt2[1]*start/dist))
        x2=int((pt1[0]*(1-stop/dist)+pt2[0]*stop/dist))
        y2=int((pt1[1]*(1-stop/dist)+pt2[1]*stop/dist))
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)
        start += (lengthDash+lengthGap)
            
def cvDrawDottedRect(img, x, y, color):

    hwidth = 20
    corner = 4
    cross = 6

#    // corners 
    
    p1=np.array((x-hwidth,y-hwidth))
    p2=np.array((x+hwidth,y-hwidth))
    p3=np.array((x+hwidth,y+hwidth))
    p4=np.array((x-hwidth,y+hwidth))
    #// draw box
    cvDrawDottedLine(img, p1, p2, color, 1, 2, 4)
    cvDrawDottedLine(img, p2, p3, color, 1, 2, 4)
    cvDrawDottedLine(img, p3, p4, color, 1, 2, 4)
    cvDrawDottedLine(img, p4, p1, color, 1, 2, 4)
    #// draw corners
    x_off=np.array((corner, 0))
    y_off=np.array((0, corner))
    cvDrawDottedLine(img, p1, p1 + x_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p1, p1 + y_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p2, p2 - x_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p2, p2 + y_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p3, p3 - x_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p3, p3 - y_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p4, p4 + x_off, color, 2, 10, 10)
    cvDrawDottedLine(img, p4, p4 - y_off, color, 2, 10, 10)
    #// draw cross
    x_coff=np.array((cross, 0))
    y_coff=np.array((0, cross))
    p1 = np.array((x,y-hwidth))
    p2 = np.array((x,y+hwidth))
    p3 = np.array((x-hwidth,y))
    p4 = np.array((x+hwidth,y))
    cvDrawDottedLine(img, p1 + y_coff, p1, color, 1, 10, 10)
    cvDrawDottedLine(img, p2, p2 - y_coff, color, 1, 10, 10)
    cvDrawDottedLine(img, p3 + x_coff, p3, color, 1, 10, 10)
    cvDrawDottedLine(img, p4 - x_coff, p4, color, 1, 10, 10)




HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'
MOVIEDIR = DATADIR + 'footage/'
df = pd.read_csv(FILELIST)

show_index =1
outputMovie=0
leaders = np.load('../../../leaders.npy')
followers = np.load('../../../followers.npy')

for index, row in df.iterrows():
    if index!=show_index:
        continue


    noext, ext = os.path.splitext(row.filename)   


    trackName =  TRACKDIR + '/RELINKED_' + str(index) + '_' + noext + '.csv' #'/TRACKS_' + str(index) + '_' + noext + '.csv'
    posName = TRACKDIR + '/CARIBOU_REAL_POS_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    movieName = MOVIEDIR + row.filename
    outputName = MOVIEDIR + '/TRACKS_' + str(index) + '_' + noext + '.avi'


    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    linkDF = pd.read_csv(trackName) 
    posDF = pd.read_csv(posName) 

    

    
    
    
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
            
        thisFrame = linkDF.ix[linkDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=6
        
        for i, trrow in thisFrame.iterrows():
            
            uid = index*10000 + int(trrow['c_id'])
            if uid in leaders:
#            cv2.putText(frame ,str(int(trrow['c_id'])) ,((int(trrow['x_px'])+12, int(trrow['y_px'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
 #           cv2.rectangle(frame, ((int( trrow['x_px'])-sz, int( trrow['y_px'])-sz)),((int( trrow['x_px'])+sz, int( trrow['y_px'])+sz)),(0,0,0),2)
                cv2.putText(frame ,'L'+str(int(trrow['c_id'])) ,((int(trrow['x_px'])+6, int(trrow['y_px'])-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(34,34,200),2)
            #cv2.rectangle(frame, ((int( trrow['x'])-sz, int( trrow['y'])-sz)),((int( trrow['x'])+sz, int( trrow['y'])+sz)),(0,0,0),2)
            #cvDrawDottedRect(frame, int( trrow['x_px']), int( trrow['y_px']),(34,34,200))
            #cvDrawDottedRect(frame, int( trrow['x_px']), int( trrow['y_px']),(0,0,0))
                cv2.circle(frame, (int(trrow['x_px']), int(trrow['y_px'])),2,(34,34,200),-1)
            if uid in followers:
                cv2.circle(frame, (int(trrow['x_px']), int(trrow['y_px'])),2,(255,255,255),-1)
                cv2.putText(frame ,'F' + str(int(trrow['c_id'])) ,((int(trrow['x_px'])+6, int(trrow['y_px'])-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,200),2)
            
        thisFrame = posDF.ix[posDF['frame']==(tt)]

        
        # draw detected objects and display
        sz=6
        
        #for i, trrow in thisFrame.iterrows():
            
       #     cv2.rectangle(frame, ((int( trrow['x_px'])-sz, int( trrow['y_px'])-sz)),((int( trrow['x_px'])+sz, int( trrow['y_px'])+sz)),(0,0,0),2)
        
        if outputMovie:
            out.write(frame)
            
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    if outputMovie:
        out.release()

