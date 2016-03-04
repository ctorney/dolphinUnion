
import os
import csv
import math
import numpy as np
from datetime import datetime

from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from math import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt



neighbours = np.load('neighbours.npy')
mvector = np.load('mvector.npy')
evector = np.load('evector.npy')
    



il=20
ia = 0.272
al = 0.0
rl = 0.88

rholist = np.arange(0,1,0.01)
probs = np.zeros_like(rholist)

for i in range(len(rholist)):

    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    be = 0.88
    
    social = 0.95
    il = 100*rholist[i]
    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    #lambdas[np.abs(mvector)>pi]=pi
    dv[np.abs(mvector)>(1-al)*pi]=pi
    dv[np.abs(mvector)<(1-al)*pi]=mvector[np.abs(mvector)<(1-al)*pi]/(1-al)
        
    # first calculate all the rhos
    rhos = np.zeros_like(neighbours[:,:,0])
    rhos[(neighbours[:,:,0]>0)&(neighbours[:,:,0]<il)&(neighbours[:,:,1]>-ia)&(neighbours[:,:,1]<ia)]=social
    
    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
    nc = np.sum(np.abs(rhos),1) # normalizing constant

    wwc = ((rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((dv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    
    wwce = (1/(2*pi)) * (1-np.power(be,2))/(1+np.power(be,2)-2*be*np.cos((dv-evector).transpose())) # weighted wrapped cauchy
    # sum along the individual axis to get the total compound cauchy
    wwc = np.sum(wwc,1)/nc
    wwc[np.isnan(wwc)]=1/(2*pi)
    
    wwc = (1.0-be)*wwc + be*wwce
    probs[i]= np.sum(np.log(wwc))
plt.figure
plt.plot(rholist,probs)