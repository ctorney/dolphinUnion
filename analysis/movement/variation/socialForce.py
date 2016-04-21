
import os
import csv
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand

import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


at_l =8.1949901467068
at_a=0.199171863411916
al_l=3.07874119229163
al_a=0.408350512048466
al_w=0.716703369139833
social=0.969100500071044
rm=0.910446683807612
re=0.963595135578525
al=0.400422492125391
be=0.124447947248256




neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector2.npy')
evector = np.load('../pdata/evector.npy')
uid = np.load('../pdata/uid.npy')
evector = evector[np.isfinite(mvector)]
uid = uid[np.isfinite(mvector)]
neighbours = neighbours[np.isfinite(mvector)]
mvector = mvector[np.isfinite(mvector)]
nonnan = np.ones_like(evector)    
nonnan[np.isnan(evector)]=0.0
evector[np.isnan(evector)]=0.0


    
n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
n_weights[neighbours[:,:,0]==0]=0.0
n_weights[neighbours[:,:,0]>at_l]=0.0
n_weights[(neighbours[:,:,1]<-at_a)|(neighbours[:,:,1]>at_a)]=0.0

na_weights = al_w*np.ones_like(neighbours[:,:,0],dtype=np.float64)
na_weights[neighbours[:,:,0]==0]=0.0
na_weights[neighbours[:,:,0]>al_l]=0.0
na_weights[(neighbours[:,:,1]<-al_a)|(neighbours[:,:,1]>al_a)]=0.0

xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1) + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1) + np.sum(np.sin(neighbours[:,:,3])*na_weights,1)

    
sv = np.empty((len(mvector),2))

sv[:,0] = xsv
sv[:,1] = ysv
    
    

svv = np.arctan2(sv[:,1],sv[:,0])
#lens = np.sqrt(sv[:,1]**2+sv[:,0]**2)
als = al*np.ones_like(svv)
als[(sv[:,1]==0)&(sv[:,0]==0)]=0
#svv = svv[(sv[:,1]!=0)|(sv[:,0]!=0)]
#mvector = mvector[(sv[:,1]!=0)|(sv[:,0]!=0)]
#evector = evector[(sv[:,1]!=0)|(sv[:,0]!=0)]
#uid = uid[(sv[:,1]!=0)|(sv[:,0]!=0)]
bes = be*nonnan
socials=social#*lens
wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
wcc = al*wcs + (1.0-al)*(be*wce+(1.0-be)*wcm)

wcc2 = (be*wce+(1.0-be)*wcm)

#social force is how much do we reduce uncertainty if we include the social vector
sf = np.log(wcc) - np.log(wcc2)


inds = np.unique(uid)
means = np.empty_like(inds)
stddevs = np.empty_like(inds)

for i,thisID in enumerate(inds):
    thisSocial=sf[uid==thisID]
    means[i]=np.mean(thisSocial)
    stddevs[i]=np.std(thisSocial)/sqrt(len(thisSocial))

x=np.argsort(means)
means = means[x]
stddevs = stddevs[x]
#stddevs=stddevs[means!=0]
#means=means[means!=0]

plt.plot(means,'r')
plt.plot(means+stddevs,'b')
plt.plot(means-stddevs,'b')
