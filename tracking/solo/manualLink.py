import numpy as np
import pandas as pd
import os

import cv2

HD = os.getenv('HOME')
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

# DROPBOX OR HARDDRIVE
#MOVIEDIR = DATADIR + 'footage/' 
MOVIEDIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'

df = pd.read_csv(FILELIST)
for index, row in df.iterrows():
    if (index!=0): continue
    noext, ext = os.path.splitext(row.filename)   
    movieName = MOVIEDIR + row.filename
    posfilename = DATADIR + 'tracked/FINAL_' + str(index) + '_' + noext + '.csv'
    outfilename = DATADIR + 'tracked/RELINKED_' + str(index) + '_' + noext + '.csv'

    posDF = pd.read_csv(posfilename) 
    
    ids = np.unique(posDF['c_id'].values)
    newids = np.unique(posDF['c_id'].values)
    trackVals = np.zeros((len(ids),6)) # start stop xstart ystart xstop ystop
    premean = 0.0
    for cnum, cpos in posDF.groupby('c_id'):
        trackVals[ids==cnum,0]=cpos['frame'].iloc[0]
        trackVals[ids==cnum,1]=cpos['frame'].iloc[-1]
        trackVals[ids==cnum,2]=cpos['x'].iloc[0]
        trackVals[ids==cnum,3]=cpos['y'].iloc[0]
        trackVals[ids==cnum,4]=cpos['x'].iloc[-1]
        trackVals[ids==cnum,5]=cpos['y'].iloc[-1]
        premean = premean +  (cpos['frame'].iloc[-1]-cpos['frame'].iloc[0])

    premean = premean/(len(ids))
    timediff = 1200 # 10 seconds
    dist = 100
    cap = cv2.VideoCapture(movieName)
    box_dim=256
    nx = 1920
    ny = 1080
    sz=16
    frName = 'is this the same caribou? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    escaped=False
    for i in range(len(ids)):
        if (i!=136): continue
        found = False
        skip=False
        for j in range(len(ids)):
            print(j)
            if i==j: continue
            if trackVals[j,0]<(trackVals[i,1]-120): continue # track j starts more than 2 secs before i finishes
            if trackVals[j,0]>(trackVals[i,1]+timediff): continue # track j starts too long after i finishes
            distance = ((trackVals[j,2]-trackVals[i,4])**2+(trackVals[j,3]-trackVals[i,5])**2)**0.5
            print(distance)
            if distance>500: continue
            print(j)
            
            
            startFrame = max(0,trackVals[i,1]-300)
            stopFrame = trackVals[j,0]+300
            cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
            ipos = posDF[posDF['c_id']==i]
            ipos = ipos[ipos['frame']>startFrame]
            jpos = posDF[posDF['c_id']==j]
            jpos = jpos[jpos['frame']<stopFrame]
            ix = ipos['x_px'].iloc[-1]
            iy = ipos['y_px'].iloc[-1]
            
            
            while True:
                thisFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if thisFrame>stopFrame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
                _, frame = cap.read()
                if (thisFrame%6) > 0 : continue
                
                allC = posDF.ix[posDF['frame']==(thisFrame)]

        
        
                for _, trrow in allC.iterrows():
            
#            cv2.putText(frame ,str(int(trrow['c_id'])) ,((int(trrow['x_px'])+12, int(trrow['y_px'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
 #           cv2.rectangle(frame, ((int( trrow['x_px'])-sz, int( trrow['y_px'])-sz)),((int( trrow['x_px'])+sz, int( trrow['y_px'])+sz)),(0,0,0),2)
            
                    cv2.circle(frame, (int(trrow['x_px']), int(trrow['y_px'])),2,(255,255,255),-1)
                for _, trow in ipos.iterrows():
                    if trow['frame']==thisFrame:
                        cv2.circle(frame, (int(trow['x_px']), int(trow['y_px'])),3,(0,0,255),-2)
                for _, trow in jpos.iterrows():
                    if trow['frame']==thisFrame:
                        cv2.circle(frame, (int(trow['x_px']), int(trow['y_px'])),4,(155,255,155),3)
                tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
                
                
       
                cv2.imshow(frName,frame)
                k = cv2.waitKey(10)
                
                if k==ord('y'):
                    found=True
                    break
                if k==ord('n'):
                    break
                if k==ord('s'):
                    skip=True
                    break
                
                if k==27:    # Esc key to stop
                    escaped=True
                    break 
            if found:
                print(str(j)+' changed to '+str(i))
                newids[ids==j]=newids[ids==i]
                break
            if escaped:
                break
            if skip:
                break
        if escaped:
            break
            
            
          
      

    cv2.destroyAllWindows()
    
    for thisID in ids:
        posDF.loc[posDF['c_id']==thisID,'c_id']=newids[ids==thisID]
    postmean = 0.0
    countID=0
    for cnum, cpos in posDF.groupby('c_id'):
        postmean = postmean +  (cpos['frame'].iloc[-1]-cpos['frame'].iloc[0])
        countID=countID+1

    postmean = postmean/(countID)
    print(str(premean/60)+" before, now: "+str(postmean/60))
    #posDF.to_csv(outfilename)
    if escaped:
        break

#    break
