

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
LOGDIR = DATADIR + '/logs/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():

    if index!=3:
        continue
    
    noext, ext = os.path.splitext(row.filename)   


    trackName = TRACKDIR + '/FINAL_' + str(index) + '_' + noext + '.csv'
    geoName = LOGDIR + '/GEOTAG_' + noext + '.csv'
    movieName = MOVIEDIR + row.filename
    

        

    # name the images after the track file name
    path, fileonly = os.path.split(trackName)
    noext, ext = os.path.splitext(fileonly)
    
    
    linkedDF = pd.read_csv(trackName) 
    geoDF = pd.read_csv(geoName) 
    nx = 1920
    ny = 1080
    
    numPars = int(linkedDF['c_id'].max()+1)
    
    caribouYN = np.zeros(shape=(0,2),dtype=int)
    box_dim = 128    
    cap = cv2.VideoCapture(movieName)
    
    sz=16
    frName = ' is caribou? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    escaped = False
    for i in range(numPars):
        if i<000:
            continue
        sys.stdout.write('\r')
        sys.stdout.write("caribou id : %d" % (i))
        sys.stdout.flush()
        #print('caribou id: ' + str(i))
        thisPar = linkedDF[linkedDF['c_id']==i]
        if escaped == True:
            break
        if thisPar.count()[0]<10:
            continue
        
#        print(thisPar.count()[0])
        for _, row in thisPar.iterrows():
    
            ix = int(row['x_px'])
            iy = int(row['y_px'])
            fNum = int(row['frame'])
            

            
            cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
            _, frame = cap.read()
            
            
            cv2.rectangle(frame, ((int( row['x_px'])-sz, int( row['y_px'])-sz)),((int( row['x_px'])+sz, int( row['y_px'])+sz)),(0,0,0),1)
            tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
            
            cv2.imshow(frName,tmpImg)
            k = cv2.waitKey(1000)
            
            if k==ord('y'):
                caribouYN = np.vstack((caribouYN, [i,1]))
                break
            if k==ord('n'):
                caribouYN = np.vstack((caribouYN, [i,0]))
                break
            if k==ord('c'):
                break
            if k==27:    # Esc key to stop
                escaped=True
                break 
            
    cv2.destroyAllWindows()
    for caribou in caribouYN:
        thisPar = linkedDF[linkedDF['c_id']==caribou[0]]
    
        for index2, row2 in thisPar.iterrows():
            
            ix = int(row2['x_px'])
            iy = int(row2['y_px'])
            fNum = int(row2['frame'])
            if (fNum%5)!=0:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
            _, frame = cap.read()
    
            # get altitude of frame
            alt = float(geoDF['alt'][fNum])/1000.0            
            # work out size of box if box if 32x32 at 100m
            grabSize = m.ceil((100.0/alt)*16.0)
            tmpImg =  cv2.cvtColor(frame[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy(), cv2.COLOR_BGR2GRAY)

            
            if tmpImg.size == 4*grabSize*grabSize:# and tmpImg[tmpImg==0].size<10 :
                if caribou[1]==1:
                    cv2.imwrite('./yes2/' + noext + '_' + str(caribou[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
                if caribou[1]==0:
                    cv2.imwrite('./no2/' + noext + '_' + str(caribou[0]) + '_' + str(fNum) + '.png',cv2.resize(tmpImg,(32,32)))
                    #break
            
    
    
    cap.release()



