
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

    

    
    
from jpype import *
import random
import math

# Change location of jar to match yours:
jarLocation = "./infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Generate some random normalised data.
numObservations = 1000
covariance=0.4
# Source array of random normals:
sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]
# Destination array of random normals with partial correlation to previous value of sourceArray
destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]], \
                                             [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
# Uncorrelated source array:
sourceArray2 = [random.normalvariate(0,1) for r in range(numObservations)]
# Create a TE calculator and run it:



    
    
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

allNeighbours = np.load('neighbours.npy')
allMvector = np.load('mvector.npy')
allEvector = np.load('evector.npy')
sin_ev = np.sin(allEvector)
cos_ev = np.cos(allEvector)

bnds = ((0.0, 1.0), (0.0, 1.0))

allIDs = np.unique(uid)

index=0
x0=(0.5,0.5)
bbb =np.mean(beta)

alss=np.arange(0,1,0.01)
vals = np.zeros(len(alss))
ind=0


#plt.plot(vals)
#

    #break
#xbins=1000
#n, xe = np.histogram(allMvector, bins=xbins)
#sy, _ = np.histogram(allMvector, bins=xbins, weights=svv)
#means = sy / n

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

teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.initialise(1, 0.01)
index=0
results = np.zeros((len(allIDs)))
lengths = np.zeros((len(allIDs)))
ids = np.zeros((len(allIDs)))
#thisID=100
for thisID in allIDs:
    
    neighbours = allNeighbours[uid==thisID]
    
    mvector = np.zeros_like(allMvector[uid==thisID])
    mvect2 = allMvector[uid==thisID]
    mvector[:-1] = mvect2[1:]
    
    if len(mvector)<5:
        continue
    #if len(mvector)>50:
    #    continue
    n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
     
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    svv = np.arctan2(ysv,xsv)
    
    
     # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    #teCalc.setObservations(JArray(JDouble, 1)(sourceArray2), JArray(JDouble, 1)(destArray))
    
    teCalc.setObservations(JArray(JDouble, 1)(svv.tolist()), JArray(JDouble, 1)(mvector.tolist()))
    # For copied source, should give something close to 1 bit:
    result = teCalc.computeAverageLocalOfObservations()
    
    lengths[index]=len(allMvector[uid==thisID])
    results[index] = lengths[index]*result
    ids[index]=thisID
    index = index+1
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
res = results[:index]
res = res[res>0]
plt.hist(res,bins=100,range=[0,20])
#results
#

thisID=60153
neighbours = allNeighbours[uid==thisID]

mvector = np.zeros_like(allMvector[uid==thisID])
mvect2 = allMvector[uid==thisID]
mvector[:-1] = mvect2[1:]

n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
 
xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
svv = np.arctan2(ysv,xsv)

thisID=10089
neighbours = allNeighbours[uid==thisID]

mvector1 = np.zeros_like(allMvector[uid==thisID])
mvect2 = allMvector[uid==thisID]
mvector1[:-1] = mvect2[1:]


n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
 
xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
svv1 = np.arctan2(ysv,xsv)
