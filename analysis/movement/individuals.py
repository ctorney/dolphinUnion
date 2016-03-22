
import os
import csv 
import math as m

from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand

import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

    
def moves(x,params):
    neighbours = params[0]
    mvector = params[1]
    evector = params[2]
    
    
    al=x[0]
    be=x[1]
    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = np.sqrt(xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    svv = np.arctan2(ysv,xsv)
    
    als = al*lens
    socials=lens*social
    wcs = (1/(2*m.pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
    wce = (1/(2*m.pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*m.pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    #print(wcs,wce,wcm)
    return -np.sum(np.log(wcc))



def socialmoves(be,params):
    neighbours = params[0]
    
    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = np.sqrt(xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    vals = be*np.sqrt(ysv**2+xsv**2)
    if len(vals[np.nonzero(vals)])==0:
        return 0
    return np.mean(vals[np.nonzero(vals)])
    
    


aa = np.load('decay_exponent.npy')
bb = np.load('interaction_length.npy')
cc = np.load('interaction_angle.npy')
social = np.mean(np.load('rho_s.npy'))
re = np.mean(np.load('rho_e.npy'))
rm = np.mean(np.load('rho_m.npy'))
alpha = np.load('alpha.npy')
beta = np.load('beta.npy')
uid = np.load('uid.npy')

de=np.mean(aa)
il=np.mean(bb)
ia = np.mean(cc)
x0 = (np.mean(alpha),np.mean(beta))
allNeighbours = np.load('neighbours.npy')
allMvector = np.load('mvector.npy')
allEvector = np.load('evector.npy')
sin_ev = np.sin(allEvector)
cos_ev = np.cos(allEvector)

bnds = ((0.0, 1.0), (0.0, 1.0))

allIDs = np.unique(uid)
results = np.zeros((len(allIDs),2))
lengths = np.zeros((len(allIDs)))
index=0
x0=(0.5,0.5)
bbb =np.mean(beta)

alss=np.arange(0,1,0.01)
vals = np.zeros(len(alss))
ind=0
for aaa in alss:
    parameters = (allNeighbours,allMvector,allEvector)
    vals[ind] = moves((aaa,bbb),parameters)
    ind=ind+1
    

plt.plot(vals)
#
#for thisID in allIDs:
#    
#    parameters = (allNeighbours[uid==thisID],allMvector[uid==thisID],allEvector[uid==thisID])
#    result = minimize(moves, x0, args=(parameters,), method='TNC',bounds=bnds, options={'maxiter':100})
#    results[index] = (result.x[0],result.x[1])
#    #results[index,1] = socialmoves(result.x[1],parameters)
#    lengths[index]=len(allMvector[uid==thisID])
#    index=index+1
#    #break
##print(moves(allNeighbours,allMvector,0,1))
#    
##plt.plot(results[:,0],results[:,1],'.')
#
#plt.hist(results[:,0],bins=200,range=(0.0,1.0))
#
##plt.hist(results[:,0],bins=100)
##plt.hist(lengths,bins=100)
#
#
