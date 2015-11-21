
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd


import pickle


from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



sys.path.append('../.')

from circularHOGExtractor import circularHOGExtractor
ch = circularHOGExtractor(4,2,4) 

fhgClass = pickle.load( open( "svmClassifier.p", "rb" ) )

#def classObjects(filename):
filename = '/home/ctorney/data/wildebeest/test.avi'
nx = 1920
ny = 1080
linkedDF = pd.read_csv('../link/output.csv') 

cap = cv2.VideoCapture(filename)
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  )
 
numPars = int(linkedDF['particle'].max()+1)    
parList = np.arange(numPars)
# column to keep track of scores
linkedDF['idCount']=0

idCount = np.zeros_like(parList)
targetScore = 5 # if we get this number of positives or negatives we assume it's been classified
for tt in range(frames):
    print(tt)
    # Capture frame-by-frame
    _, frame = cap.read()
    if (tt%15) > 0 : continue
    thisFrame = linkedDF.ix[linkedDF['frame']==tt]

    
    # draw detected objects and display
    sz=16
    
    for i, row in thisFrame.iterrows():
    
        ix = int(row['x'])
        iy = int(row['y'])
        pid = int(row['particle'])
        sc = int(row['idCount'])
        if (sc>=targetScore):
            continue
                

        
        tmpImg =  cv2.cvtColor(frame[max(0,iy-sz):min(ny,iy+sz), max(0,ix-sz):min(nx,ix+sz)].copy(), cv2.COLOR_BGR2GRAY)
        if tmpImg.size == 4*sz*sz and tmpImg[tmpImg==0].size<10 :
            res = fhgClass.predict(ch.extract(tmpImg))
            
            if res[0]>0.5:
                linkedDF.loc[linkedDF['particle']==pid,'idCount']=sc + 1

                #linkedDF['idCount'][linkedDF['particle']==pid].value= sc + 1
            else:
                linkedDF.loc[linkedDF['particle']==pid,'idCount']= sc - 1
        
    # now erase rows for which we've had over targetScore negatives
    linkedDF = linkedDF[linkedDF['idCount']>-targetScore]

# erase rows for which had a non-positive score
linkedDF = linkedDF[linkedDF['idCount']>0]       
    

linkedDF.to_csv('output.csv')
cv2.destroyAllWindows()
cap.release()

if __name__ == '__main__':
    FULLNAME = '/home/ctorney/data/wildebeest/test.avi'
    classObjects(FULLNAME)
