# transfer entropy from movement to social vector as calculated from Bayesian inference

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
    
from jpype import *

# Change location of jar to match yours:
jarLocation = "./infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
#startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


X_plot = np.linspace(0, 2, 1000)[:, np.newaxis]

uid = np.load('../uid.npy')    
allNeighbours = np.load('../neighbours.npy')
allMvector = np.load('../mvector.npy')
allEvector = np.load('../evector.npy')

allIDs = np.unique(uid)
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.initialise(1, 0.25)

index=0
results = np.zeros((len(allIDs)))
lengths = np.zeros((len(allIDs)))
ids = np.zeros((len(allIDs)))

#parameters from inference
ilc=14.1455
iac=0.2210
    
np.random.seed(100)
for thisID in allIDs:
    
    neighbours = allNeighbours[uid==thisID]
    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
    n_weights[neighbours[:,:,0]==0]=0.0
    n_weights[neighbours[:,:,0]>ilc]=0.0
    n_weights[(neighbours[:,:,1]<-iac)|(neighbours[:,:,1]>iac)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    
    svv = np.arctan2(ysv,xsv)
    
    mvect2 = allMvector[uid==thisID]
 #   mvect2 = mvect2[(xsv!=0)|(ysv!=0)]
#    svv = svv[(xsv!=0)|(ysv!=0)]
    #np.random.uniform(-iac,iac,2)
    svv[(xsv==0)&(ysv==0)]=np.random.uniform(-iac,iac,np.size(svv[(xsv==0)&(ysv==0)]))

    mvector = np.zeros_like(mvect2)
    mvector[:-1] = mvect2[1:]
    lengths[index]=len(allMvector[uid==thisID])
   
    # 1 minute tracks used only
    if len(mvect2)<30:
        continue
    ids[index]=thisID
    index = index+1
     
    teCalc.setObservations(JArray(JDouble, 1)(svv.tolist()), JArray(JDouble, 1)(mvector.tolist()))
    result = teCalc.computeAverageLocalOfObservations()
    results[index-1] = result
res = (results[:index])
ids = (ids[:index])

kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(res[:,np.newaxis])
plt.hist(res,bins=25,range=[0,2],normed=True,alpha=0.5)
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'k-',linewidth=2)
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4)#, fc='#AAAAFF')
plt.savefig('te.png')


for idx,val in enumerate(log_dens):
    if idx==0:
        prev = val
        prevInc=True
        continue
    if val>prev:
        increasing=True
    else:
        increasing=False
    if prevInc!=increasing:
        if increasing:
            threshold = X_plot[idx]
    prevInc = increasing
    prev=val

#threshold=0.5786
leaders = ids[res<threshold]
followers = ids[res>threshold]
#np.save('../../leaders2.npy',leaders)
#np.save('../../followers2.npy',followers)