
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

from sklearn.neighbors import KernelDensity
X_plot = np.linspace(0, 2, 1000)[:, np.newaxis]
    
teCalc.initialise(1, 0.25)
index=0
results = np.zeros((len(allIDs)))
lengths = np.zeros((len(allIDs)))
ids = np.zeros((len(allIDs)))
#thisID=100
ia=0.3007855
de=0.2
il=15
allSVV = np.array([])
allSVV2 = np.array([])
for thisID in allIDs:
    
    neighbours = allNeighbours[uid==thisID]
    
    #if len(mvector)>50:
    #    continue
#    n_weights=np.zeros_like(neighbours[:,:,0])
#    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
#    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
#    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
#    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
#    svv = np.arctan2(ysv,xsv)
#    
#    allSVV = np.append(allSVV,svv)
    
    n_weights = np.exp(-neighbours[:,:,0]/il)
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,0]==0]=0.0
    nn = np.argmax(n_weights, axis=1)
    xsv2 = np.zeros_like(nn,dtype=np.float64)
    ysv2 = np.zeros_like(nn,dtype=np.float64)
    svv2 = np.zeros_like(nn,dtype=np.float64)

    for idx, val in enumerate(nn):
        if n_weights[idx,val]>0:
            xsv2[idx] = np.cos(neighbours[idx,val,1])*(neighbours[idx,val,0])
            ysv2[idx] = np.sin(neighbours[idx,val,1])*(neighbours[idx,val,0])
            #svv2[idx] = neighbours[idx,val,1]
    svv2 = np.arctan2(ysv2,xsv2)
    
    allSVV2 = np.append(allSVV2,svv2)
   # svv2 = svv2[(xsv2!=0)|(ysv2!=0)]
    #check=svv2[svv2>ia]
    #if len(check):
    #    break
    
    #n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))*(0.5+0.5*np.tanh(ad*(ia-np.abs(neighbours[:,:,1]))))

    
    
    
    mvect2 = allMvector[uid==thisID]
    if len(mvect2)<25:
        continue
    mvect2 = mvect2[(xsv2!=0)|(ysv2!=0)]
    print(len(mvect2))
    svv = svv2[(xsv2!=0)|(ysv2!=0)]
    mvector = np.zeros_like(mvect2)
    mvector[:-1] = mvect2[1:]
    lengths[index]=len(allMvector[uid==thisID])
   
    
    if len(mvect2)<25:
        continue
    ids[index]=thisID
    index = index+1
     # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    #teCalc.setObservations(JArray(JDouble, 1)(sourceArray2), JArray(JDouble, 1)(destArray))
    
    teCalc.setObservations(JArray(JDouble, 1)(svv.tolist()), JArray(JDouble, 1)(mvector.tolist()))
    # For copied source, should give something close to 1 bit:
    result = teCalc.computeAverageLocalOfObservations()
    results[index-1] = result#*lengths[index]
    
    #if len(mvector)==19:
     #   break

#print(result)
   
#plt.hist(results[:index],bins=500)

#
#plt.plot(lengths[:index],lengths[:index]*results[:index],'.')
#innd = 0
#for xy in zip(lengths[:index],results[:index]):                                       
#    if (ids[innd]==60153): 
#        print(ids[innd])
#        innd=innd+1
#        continue
##    plt.annotate('(%s)' % ids[innd], xy=xy, textcoords='data') 
#    innd=innd+1
#plt.annotate(ids[:index], xy = (lengths[:index],results[:index]), xytext = (0, 0), textcoords = 'offset points')
#plt.figure
res = (results[:index])
#res = res[res>0]

kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(res[:,np.newaxis])
plt.hist(res,bins=25,range=[0,2],normed=True,alpha=0.5)
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'k-',linewidth=2)
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.6)#, fc='#AAAAFF')
plt.savefig('two_groups.png')
xbins=20
n, xe = np.histogram(lengths[:index], bins=xbins)
sy, _ = np.histogram(lengths[:index], bins=xbins, weights=results[:index])
means = sy / n
#plt.plot(xe[:-1],means)
