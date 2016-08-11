
import os
import csv
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand

from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def networkP():

    neighbours = np.load('../pdata/neighbours.npy')
    mvector = np.load('../pdata/mvector.npy')
    evector = np.load('../pdata/evector.npy')


    netcount=1

# variable to normalize the move step lengths for the alignment rule
    dists = neighbours[:,:,4]
    stepLen=np.mean(dists[dists>0])

    social =0.9387376717386503
    al =0.23569999931373675
    be =  0.1442806482086779

    il = 1
    ia = 0.2842569047789764
    al_w = 1.00182262664265
    ig = 1.316401214251669 
    social = 0.9396529725088002
    al = 0.3870379791643967
    be = 0.13673084835299582

#    social = 0.9396529725088002
#    al = 0.3870379791643967
#    be = 0.13673084835299582
#    ig = 1.3065421032118985
#    al_w = 0.7659181663834961
 #   ia=0.27642330874196225
    #at_de = 9.272935648814755
    #at_l = 4.994414655997313
    #at_a=0.27642330874196225
    #ig = 1.3065421032118985
    #al_w = 0.7659181663834961






    

#xj = (neighbours[:,:,0]*np.cos(neighbours[:,:,1]))+(np.cos(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])
#yj = (neighbours[:,:,0]*np.sin(neighbours[:,:,1]))+(np.sin(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])
    xj = (neighbours[:,:,0]*np.cos(neighbours[:,:,1]))+(np.cos(neighbours[:,:,3])*(al_w))#/stepLen)*neighbours[:,:,4])
    yj = (neighbours[:,:,0]*np.sin(neighbours[:,:,1]))+(np.sin(neighbours[:,:,3])*(al_w))#/stepLen)*neighbours[:,:,4])
        
    anglesj=np.arctan2(yj,xj)
        
    distances = neighbours[:,:,0].copy()
    distances[(neighbours[:,:,0]<=ig)]=9999.0
    distances[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=9999.0
    networkDist = np.argsort(distances,axis=1).astype(np.float32)
    networkDist = np.argsort(networkDist,axis=1).astype(np.float32)

    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
    n_weights[(networkDist)>=netcount]=0.0
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[(neighbours[:,:,0]<=ig)]=0.0

#   na_weights = al_w*np.ones_like(neighbours[:,:,0],dtype=np.float64)
#   na_weights[(networkDist)>=il]=0.0
#   na_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
#   na_weights[neighbours[:,:,0]==0]=0.0
    
#xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1) + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
#    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1) + np.sum(np.sin(neighbours[:,:,3])*na_weights,1)
    
   
 
    xsv = np.sum(np.cos(anglesj)*n_weights,1)
    ysv = np.sum(np.sin(anglesj)*n_weights,1)
    
    sv = np.empty((len(mvector),2))

    sv[:,0] = xsv
    sv[:,1] = ysv
    



    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    als = al*np.ones_like(svv)
    als[(sv[:,1]==0)&(sv[:,0]==0)]=0
    xvals = als*np.cos(svv) + (1.0-als)*(be*np.cos(evector)+(1.0-be))
    yvals = als*np.sin(svv) + (1.0-als)*(be*np.sin(evector))

    allV = np.arctan2(yvals,xvals)
    
    wcs = (1/(2*pi)) * (1-(social*social))/(1+(social*social)-2*social*np.cos((allV-mvector).transpose()))

    return ((wcs))
