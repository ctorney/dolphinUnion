
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


allNeighbours= np.load('neighbours.npy')
allMvector = np.load('mvector.npy')
allEvector = np.load('evector.npy')
    

aa = np.load('decay_exponent.npy')
bb = np.load('interaction_length.npy')
cc = np.load('interaction_angle.npy')
social = np.mean(np.load('rho_s.npy'))
re = np.mean(np.load('rho_e.npy'))
rm = np.mean(np.load('rho_m.npy'))
de=np.mean(aa)
il=np.mean(bb)
ia = np.mean(cc)
be = np.mean(np.load('beta.npy'))
uid = np.load('uid.npy')


rholist = np.arange(0.01,1.0,0.01)
probs = np.zeros_like(rholist)
thisID = 2
for thisID2 in range(1):
    thisID=3
    neighbours = allNeighbours[uid==thisID]
    mvector =allMvector[uid==thisID]
    evector =allEvector[uid==thisID]
    
    for i in range(len(rholist)):
    
        # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
        # and the assumed interaction function
    
        al = rholist[i]
        
    #    social = rholist[i]#0.95
     #   il =10# 100*rholist[i]
        dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
        #lambdas[np.abs(mvector)>pi]=pi
            
        
        n_weights = ((neighbours[:,:,0]/il)*np.exp((1.0/de)*(1.0-(neighbours[:,:,0]/il)**de)))
        n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
     
        xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
        ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
        
        lens = np.sqrt(xsv**2+ysv**2)
        ysv[lens>1]=ysv[lens>1]/lens[lens>1]
        xsv[lens>1]=xsv[lens>1]/lens[lens>1]
        svv = np.arctan2(ysv,xsv)
        lens = np.sqrt(xsv**2+ysv**2)
        als = al*lens
        socials=lens*social
        wcs = (1/(2*m.pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
        wce = (1/(2*m.pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
        wcm = (1/(2*m.pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
        #print(np.sum(np.log(wcs)))
        wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
        probs[i]= np.sum(np.log(wcc))
    plt.figure
    plt.plot(rholist,probs)