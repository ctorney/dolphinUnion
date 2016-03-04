
import cv2
import numpy as np
import pandas as pd
import os
import re
import math
import time

HD = os.getenv('HOME')
DD = '/media/ctorney/SAMSUNG/'

DATADIR = DD + ''
CLIPDIR = DD + ''
CLIPLIST = HD + '/workspace/CBC/cliplist/'



df = pd.read_csv(CLIPLIST)
df['clipname']=''




for index, row in df.iterrows():
    

    h,m,s = re.split(':',row.start)
    timeStart = int(h)*3600+int(m)*60+int(s)
    h,m,s = re.split(':',row.stop)
    timeStop = int(h)*3600+int(m)*60+int(s)

    inputName = DATADIR + row.folder + '/' + row.filename
    outputName = time.strftime("%Y%m%d", time.strptime(row.date,"%d-%b-%Y")) + '-' + str(index) + '.avi'

    df.loc[index,'clipname'] = outputName
    
    
    print('Movie ' + row.folder + '/' + row.filename + ' from ' + str(timeStart) + ' to ' + str(timeStop) + ' out to ' + outputName)
#   if index<6: continue
#    if index>10: continue

    cap = cv2.VideoCapture(inputName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    fStart = timeStart*fps
    fStop = timeStop*fps

    cap.set(cv2.CAP_PROP_POS_FRAMES,fStart)
    S = (1920,1080)
    
    # reduce to ten frames a second
    ds = math.ceil(fps/10)
    out = cv2.VideoWriter(CLIPDIR + outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
    for tt in range(fStart,fStop):
        # Capture frame-by-frame
        _, frame = cap.read()
        if (tt%ds!=0):
            continue
        out.write(frame)
        

    cap.release()
    out.release()
    
#df.to_csv(CLIPLIST,index=False)


