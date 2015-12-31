

import cv2
import numpy as np
import os,sys
import re
import math 
import pandas as pd
import random


HD = os.getenv('HOME')

MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'


MOVIEDIR = HD + '/Dropbox/presentations/movies/caribou/'



movieName = MOVIEDIR + 'caribou_pan2.mp4'







cap = cv2.VideoCapture(movieName)
fps = round(cap.get(cv2.CAP_PROP_FPS))

fStart = 0*fps
fStop = 43*fps


cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
S = (1280,720)


out = cv2.VideoWriter('tmp'+str(random.randint(0,10000))+ '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)

for tt in range(fStart,fStop):

    _, frame = cap.read()
    frame[-47:-1,:,:]=0
        
    #frame = np.roll(frame, 47, 0)
    frame = np.roll(frame, 30, 0)
    
    if 1:
        out.write(frame)
        
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()

