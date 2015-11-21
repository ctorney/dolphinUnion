

import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd


def createYesNoSamples(filename):
    #filename='../../test.avi'
    path, fileonly = os.path.split(filename)
    noext, ext = os.path.splitext(fileonly)
    
    
    linkedDF = pd.read_csv('../../link/output.csv') 
    nx = 1920
    ny = 1080
    
    numPars = int(linkedDF['particle'].max()+1)
    
    
    box_dim = 128    
    cap = cv2.VideoCapture(filename)
       
    sz=16
    frName = ' is wildebeest? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    for i in range(numPars):
        thisPar = linkedDF[linkedDF['particle']==i]
        
        for _, row in thisPar.iterrows():
    
            ix = int(row['x'])
            iy = int(row['y'])
            fNum = int(row['frame'])
            
            cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
            _, frame = cap.read()
            
            
            cv2.rectangle(frame, ((int( row['x'])-sz, int( row['y'])-sz)),((int( row['x'])+sz, int( row['y'])+sz)),(0,0,0),1)
            tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
            
            cv2.imshow(frName,tmpImg)
            k = cv2.waitKey(1000)
            
            if k==ord('y'):
                for _, row2 in thisPar.iterrows():
                    ix = int(row2['x'])
                    iy = int(row2['y'])
                    fNum = int(row2['frame'])
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
                    _, frame = cap.read()
            
                    tmpImg =  cv2.cvtColor(frame[max(0,iy-sz):min(ny,iy+sz), max(0,ix-sz):min(nx,ix+sz)].copy(), cv2.COLOR_BGR2GRAY)
                    if tmpImg.size == 4*sz*sz and tmpImg[tmpImg==0].size<10 :
                        cv2.imwrite('./yes/' + noext + '_' + str(i) + '_' + str(fNum) + '.png',tmpImg)
                break
            if k==ord('n'):
                for _, row2 in thisPar.iterrows():
                    ix = int(row2['x'])
                    iy = int(row2['y'])
                    fNum = int(row2['frame'])
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fNum)
                    _, frame = cap.read()
            
                    tmpImg =  cv2.cvtColor(frame[max(0,iy-sz):min(ny,iy+sz), max(0,ix-sz):min(nx,ix+sz)].copy(), cv2.COLOR_BGR2GRAY)
                    if tmpImg.size == 4*sz*sz and tmpImg[tmpImg==0].size<10 :
                        cv2.imwrite('./no/' + noext + '_' + str(i) + '_' + str(fNum) + '.png',tmpImg)
                break
            if k==27:    # Esc key to stop
                cv2.destroyAllWindows()
                cap.release()
                return
    
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    #FULLNAME = sys.argv[1]
    FULLNAME = '/home/ctorney/data/wildebeest/test.avi'

    createYesNoSamples(FULLNAME)

