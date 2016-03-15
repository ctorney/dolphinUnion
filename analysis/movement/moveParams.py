
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
ia = 0.8
al = 0.0
rl = 0.88
ig=1


rholist = np.arange(0.01,1,0.01)
probs = np.zeros_like(rholist)

for i in range(len(rholist)):

    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    be = 0.92
    alpha = 0.5
    
    social = rholist[i]#0.95
    il =10# 100*rholist[i]
    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    #lambdas[np.abs(mvector)>pi]=pi
        
    
    n_weights = np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
    n_weights[(neighbours[:,:,0]==0)|(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    #n_weights = np.exp(-np.abs(neighbours[:,:,1])/ia)*np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
    #n_weights[(neighbours[:,:,0]==0)]=0.0
    
    xpos = np.cos(neighbours[:,:,1])*n_weights
    ypos = np.sin(neighbours[:,:,1])*n_weights
    
    sv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
    ysv = np.sin(sv)
    xsv = np.cos(sv)
    xsv[(np.sum(ypos,1)==0)&(np.sum(xpos,1)==0)] = 0.0
    ally = be*ysv+(1.0-be)*(1.0-alpha)*np.sin(evector)
    allx = be*xsv+(1.0-be)*(alpha*np.ones_like(mvector)+(1.0-alpha)*np.cos(evector))
    dv = np.arctan2(ally,allx)
    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
    #nc = np.sum(np.abs(rhos),1) # normalizing constant

    #wwc = ((rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((dv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy
    probs[i]= np.sum(np.log(wcs))
plt.figure
plt.plot(rholist,probs)