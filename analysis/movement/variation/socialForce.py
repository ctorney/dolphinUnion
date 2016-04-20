
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


at_l = 8.543706399430775
at_a=0.1981991683183868
al_l=2.786165151495565
al_a=0.41132367012957816
al_w=0.719445094861646
social=0.9679783684078092
rm=0.9182859451906178
re=0.9207242019320199
al=0.40091298510428786
be=0.13242601151232752



neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector2.npy')
evector = np.load('../pdata/evector.npy')
evector = evector[np.isfinite(mvector)]
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
bes = be*nonnan
socials=social#*lens
wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
wcc = als*wcs + (1.0-als)*(bes*wce+(1.0-bes)*wcm)

wcc2 = (bes*wce+(1.0-bes)*wcm)

#social force is how much do we reduce uncertainty if we include the social vector
sf = np.log(wcc) - np.log(wcc2)


