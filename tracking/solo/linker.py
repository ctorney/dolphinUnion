
import numpy as np
import pandas as pd
import os
from scipy import interpolate
import trackpy as tp

import trackpy.predict

HD = os.getenv('HOME')

#DATADIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():
    if index!=3:
        continue

    noext, ext = os.path.splitext(row.filename)   
    
    # ********* read in log file and extract position data ********* 

    posName = TRACKDIR + '/CARIBOU_REAL_POS_' + str(index) + '_' + noext + '.csv'
    trackName = TRACKDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    
    toLink = pd.read_csv(posName,index_col=0)
    pred = trackpy.predict.NearestVelocityPredict()
 #   pred = trackpy.predict.NearestVelocityPredict(span=30)



  #  t = tp.link_df(toLink,15,memory=30)


    f_iter = (frame for fnum, frame in toLink.groupby('frame'))
    t = pd.concat(pred.link_df_iter(f_iter, 10.00, memory=90))
    
    outTracks = pd.DataFrame(columns= ['frame','x','y','x_px','y_px','c_id'])
    minFrames = 300
    for cnum, cpos in t.groupby('particle'):
        
        frameLen = max(cpos['frame'])-min(cpos['frame'])
        if frameLen<minFrames:
            continue
        # interpolate to smooth and fill in any missing frames        
        frameTimes = np.arange(min(cpos['frame'])+6,max(cpos['frame']),6)  
        posData = cpos[['x','y','x_px','y_px']].values
        timeData= cpos[['frame']].values
        tData = interpolate.interp1d(timeData[:,0], posData.T)(frameTimes)
        tData=tData.T

    
        
        newcpos = pd.DataFrame(np.column_stack((frameTimes,tData)), columns= ['frame','x','y','x_px','y_px'])  
        newcpos['c_id']=cnum
        
        outTracks = outTracks.append(newcpos,ignore_index=True )



        

    outTracks.to_csv(trackName, index=False)    
    
    #t.to_csv(trackName, index=False)
