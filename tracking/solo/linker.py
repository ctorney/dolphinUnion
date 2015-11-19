import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import trackpy as tp

import trackpy.predict

toLink = pd.read_csv('./TRACKS_p2_006.csv',index_col=0)
#pred = tp.predict.NearestVelocityPredict()
pred = trackpy.predict.NearestVelocityPredict()




f_iter = (frame for fnum, frame in toLink.groupby('frame'))
t = pd.concat(pred.link_df_iter(f_iter, 10.5, memory=2))


#t = tp.link_df(toLink, 8, memory=3)
t1 = tp.filtering.filter_stubs(t,150)

t1.to_csv('output.csv')
