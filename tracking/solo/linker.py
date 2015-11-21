
import numpy as np
import pandas as pd
import os

import trackpy as tp

import trackpy.predict

HD = os.getenv('HOME')

#DATADIR = '/media/ctorney/SAMSUNG/data/dolphinUnion/solo/'
DATADIR = HD + '/Dropbox/dolphin_union/2015_footage/Solo/'
TRACKDIR = DATADIR + '/tracked/'
FILELIST = HD + '/workspace/dolphinUnion/tracking/solo/fileList.csv'

df = pd.read_csv(FILELIST)

for index, row in df.iterrows():

    noext, ext = os.path.splitext(row.filename)   
    
    # ********* read in log file and extract position data ********* 

    posName = TRACKDIR + '/POSITIONS_' + str(index) + '_' + noext + '.csv'
    trackName = TRACKDIR + '/TRACKS_' + str(index) + '_' + noext + '.csv'
    
    toLink = pd.read_csv(posName,index_col=0)
    pred = trackpy.predict.NearestVelocityPredict()




    f_iter = (frame for fnum, frame in toLink.groupby('frame'))
    t = pd.concat(pred.link_df_iter(f_iter, 10.5, memory=2))


    t1 = tp.filtering.filter_stubs(t,2)

    t1.to_csv(trackName)
