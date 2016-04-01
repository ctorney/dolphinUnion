
import os
import csv
import math
import numpy as np
from datetime import datetime
import math as m
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from math import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt



social=0.9622
il = 14.1531
ia=0.2208
rm=0.9187
re = 0.9524
al = 0.3874
be = 0.1342
socialL=0.9455
rho_mL=0.9165
rho_eL=0.9412
alphaL=0.2727
betaL =0.1970

al = alphaL
be = betaL
rm = rho_mL
re = rho_eL

neighbours = np.load('../neighbours.npy')
mvector = np.load('../mvector.npy')
evector = np.load('../evector.npy')
uid = np.load('../uid.npy')

uniID = np.unique(uid)
leaders = np.load('../../leaders.npy')
#leaders= np.random.choice(uniID,len(leaders),replace=False)
# find all non-leaders
leadIndexes=(np.in1d(uid,leaders))
#neighbours= neighbours[leadIndexes]
mvector = mvector[leadIndexes]
#evector = evector[leadIndexes]


leadIndexes=(np.in1d(neighbours[:,:,2],leaders).reshape(neighbours[:,:,2].shape))
lead_weights = np.zeros_like(neighbours[:,:,2],dtype=np.float64)
lead_weights[leadIndexes]=1.0

lwlist = np.arange(1,3,0.1)
probs=np.zeros(len(lwlist))
for i in range(len(lwlist)):
    
        
    lw = lwlist[i]
        
    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)+lw*lead_weights
    n_weights[neighbours[:,:,0]==0]=0.0
    n_weights[neighbours[:,:,0]>il]=0.0
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    #lens = np.sqrt(xsv**2+ysv**2)
    #ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    #xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    sv = np.empty((len(mvector),2))

    sv[:,0] = xsv
    sv[:,1] = ysv
    


    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    #lens = np.sqrt(sv[:,1]**2+sv[:,0]**2)
    als = al*np.ones_like(svv)
    als[(sv[:,1]==0)&(sv[:,0]==0)]=0
    socials=social#*lens
    wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
    wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    
    probs[i]= np.sum(np.log(wcc))
plt.figure
plt.plot(lwlist,probs)
plt.xlabel('increased leader weighting')
plt.ylabel('log likelihood')
#plt.xlim((5,9))
plt.savefig('lbonusL.png')