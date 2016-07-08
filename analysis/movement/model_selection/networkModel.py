
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

__all__ = ['netcount','ignore_length','interaction_length','interaction_angle','rho_s','rho_m','rho_e','alpha','beta','mvector']


ignore_length = Uniform('ignore_length', lower=0.0, upper=5.0,value=0.72)
interaction_length = Uniform('interaction_length', lower=0, upper=20)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi,value=0.18)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.968)
alpha = Uniform('alpha',lower=0, upper=1,value=0.30)

rho_m = 0.937
rho_e = 0.956
beta = 0.126

neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')


netcount=1
    
@deterministic(plot=False)
def social_vector(il=interaction_length, ia=interaction_angle, ig=ignore_length):
        
    distances = neighbours[:,:,0].copy()
    distances[(neighbours[:,:,0]<=ig)]=9999.0
    distances[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=9999.0
    networkDist = np.argsort(distances,axis=1).astype(np.float32)
    networkDist = np.argsort(networkDist,axis=1).astype(np.float32)

    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
    n_weights[(networkDist)>=netcount]=0.0
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[(neighbours[:,:,0]<=ig)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out



@stochastic(observed=True)
def moves(social=rho_s, rm=rho_m,re=rho_e,al=alpha, be=beta, sv=social_vector, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    als = al*np.ones_like(svv)
    als[(sv[:,1]==0)&(sv[:,0]==0)]=0
    
    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
    wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    wcc = wcc[wcc>0]
    return np.sum(np.log(wcc))
