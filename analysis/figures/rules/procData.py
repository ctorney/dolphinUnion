import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
from math import pi
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from viridis import viridis
from scipy.stats import binned_statistic_2d
from scipy.stats import norm


HD = os.getenv('HOME')
MOVEFILE = HD + '/workspace/dolphinUnion/analysis/movement/pdata/mvector.npy'
NEIGHBOURFILE = HD + '/workspace/dolphinUnion/analysis/movement/pdata/neighbours.npy'
moves = np.load(MOVEFILE)
neighbours = np.load(NEIGHBOURFILE)
al_w=0.75
dists = neighbours[:,:,4]
stepLen=np.mean(dists[dists>0])

xj = (neighbours[:,:,0]*np.cos(neighbours[:,:,1]))+(np.cos(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])
yj = (neighbours[:,:,0]*np.sin(neighbours[:,:,1]))+(np.sin(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])


mxj = np.mean(xj[(xj!=0)|(yj!=0)])
myj = np.mean(yj[(xj!=0)|(yj!=0)])

mx = np.cos(moves)
my = np.sin(moves)
mmx = np.mean(mx)
mmy = np.mean(my)

mx = mx - mmx
my = my - mmy
angles = np.arctan2(my,mx)
mx = np.cos(angles)
my = np.sin(angles)


xj = xj - mxj
yj = yj - myj

angles = np.arctan2(yj,xj)
xj = np.cos(angles)
yj = np.sin(angles)

correlations = np.zeros((len(moves),3))
#dps = math.sqrt(((xj-mxj)*(mx-mmx))**2+((yj-myj)*(my-mmy))**2)
for i, _ in enumerate(moves):
    nrow = neighbours[i,:,:]
    print(i)
    for n, _ in enumerate(nrow):
        if neighbours[i,n,0]>0:
            thisDP = math.sqrt(((xj[i,n])*(mx[i]))**2+((yj[i,n])*(my[i]))**2)
            correlations[i,:]=[neighbours[i,n,0],neighbours[i,n,1],thisDP]

np.save('correlations.npy',correlations)
