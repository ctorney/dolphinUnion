
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


at_l = 6.6710
at_a=0.2111
al_w=0.1350
social=0.9657
al=0.4773 
be=0.136 
rm = 0.92
re = 0.93






neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')
uid = np.load('../pdata/uid.npy')


    
n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
n_weights[neighbours[:,:,0]==0]=0.0
n_weights[neighbours[:,:,0]>at_l]=0.0
n_weights[(neighbours[:,:,1]<-at_a)|(neighbours[:,:,1]>at_a)]=0.0

#   na_weights = al_w*np.ones_like(neighbours[:,:,0],dtype=np.float64)
#   na_weights[neighbours[:,:,0]==0]=0.0
#   na_weights[neighbours[:,:,0]>al_l]=0.0
#   na_weights[(neighbours[:,:,1]<-al_a)|(neighbours[:,:,1]>al_a)]=0.0

#xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1) + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
#   ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1) + np.sum(np.sin(neighbours[:,:,3])*na_weights,1)
xsv = np.sum((np.cos(neighbours[:,:,1])+np.cos(neighbours[:,:,3])*al_w) *n_weights,1)# + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
ysv = np.sum((np.sin(neighbours[:,:,1])+np.sin(neighbours[:,:,3])*al_w) *n_weights,1)#ysv




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

socials=social#*lens
wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)



asocwcc = (be*wce+(1.0-be)*wcm)



wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((0))) # weighted wrapped cauchy
maxswcc = al*wcs + (1.0-al)*(be*wce+(1.0-be)*wcm)

wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((-pi))) # weighted wrapped cauchy
minswcc = al*wcs + (1.0-al)*(be*wce+(1.0-be)*wcm)


#social force is how much do we reduce uncertainty if we include the social vector
sf = np.log(wcc) - np.log(asocwcc)
maxSF = np.log(maxswcc) - np.log(asocwcc)
minSF = np.log(minswcc) - np.log(asocwcc)

#sf = (sf-minSF)/(maxSF-minSF)

inds = np.unique(uid)
means = np.zeros_like(inds)
stddevs = np.zeros_like(inds)
newID = np.zeros_like(uid)

for i,thisID in enumerate(inds):
    thisSocial=sf[uid==thisID]
    #newID[uid==thisID]=i
 #   if len(thisSocial)<2:
 #       continue
    #print(len(thisSocial))
    means[i]=np.mean(thisSocial[thisSocial!=0])
    stddevs[i]=np.std(thisSocial)/sqrt(len(thisSocial))

x=np.argsort(means)
means = means[x]
stddevs = stddevs[x]
stddevs=stddevs[means!=0]
means=means[means!=0]
inds = inds[x]

sfSORT = np.array([])
iSORT = np.array([])
ii = 0
for i,thisID in enumerate(inds):
    thisSocial=sf[uid==thisID]
    #if len(thisSocial)<5:
    #    continue
    if np.isnan(means[i]):
        continue
    IDlist = np.ones_like(thisSocial[thisSocial!=0])*ii
    sfSORT = np.hstack((sfSORT,thisSocial[thisSocial!=0]))
    iSORT = np.hstack((iSORT,IDlist))
    ii = ii + 1


plt.plot(means,'r')
#plt.plot(means+stddevs,'b')
#plt.plot(means-stddevs,'b')
#plt.figure()
plt.plot(iSORT,sfSORT,'.')
empties = np.zeros_like(inds)
for i,thisID in enumerate(inds):
    if np.isnan(means[i]):
        continue
    thisSocial=sv[uid==thisID]
    empties[i]=len(thisSocial[(thisSocial[:,1]==0)&(thisSocial[:,0]==0)])/len(thisSocial)
empties = empties[np.isfinite(means)]
plt.plot(empties,'.')



[counts,_,_] = np.histogram2d(sfSORT,iSORT,bins=40)
counts = counts/np.sum(counts,0)
plt.figure()
plt.pcolormesh(counts,vmin=0,vmax=0.15,cmap='bone')#,lw=0.0,vmin=np.min(hista2),vmax=np.max(hista2),cmap='viridis')
uid = np.load('../pdata/uid.npy')



