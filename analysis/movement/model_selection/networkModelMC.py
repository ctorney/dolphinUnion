
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

__all__ = ['ignore_length','interaction_length','interaction_angle','rho_s','alpha','beta','mvector']


ignore_length = Uniform('ignore_length', lower=0.0, upper=5.0,value=1.3)
interaction_length = DiscreteUniform('interaction_length', lower=0, upper=20)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi,value=0.25)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.939)
alpha = Uniform('alpha',lower=0, upper=1,value=0.357)
beta = Uniform('beta',lower=0, upper=1,value=0.14)


neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')


netcount=0
    
@deterministic(plot=False)
def social_vector(il=interaction_length, ia=interaction_angle, ig=ignore_length):
        
    distances = neighbours[:,:,0].copy()
    distances[(neighbours[:,:,0]<=ig)]=9999.0
    distances[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=9999.0
    networkDist = np.argsort(distances,axis=1).astype(np.float32)
    networkDist = np.argsort(networkDist,axis=1).astype(np.float32)

    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
    n_weights[(networkDist)>=il]=0.0
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[(neighbours[:,:,0]<=ig)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out

@stochastic(observed=True)
def moves(social=rho_s, al=alpha,be=beta,sv=social_vector, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    als = al*np.ones_like(svv)
    als[(sv[:,1]==0)&(sv[:,0]==0)]=0
    xvals = als*np.cos(svv) + (1.0-als)*(be*np.cos(evector)+(1.0-be))
    yvals = als*np.sin(svv) + (1.0-als)*(be*np.sin(evector))

    allV = np.arctan2(yvals,xvals)
    
    wcs = (1/(2*pi)) * (1-(social*social))/(1+(social*social)-2*social*np.cos((allV-mvector).transpose()))

    wcs = wcs[wcs>0]
    return np.sum(np.log(wcs))


