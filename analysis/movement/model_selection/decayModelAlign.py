
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

__all__ = ['ignore_length','attract_exponent','attract_length','attract_angle','align_weight','rho_s','rho_m','rho_e','alpha','beta','mvector']


align_weight = Uniform('align_weight', lower=0.0, upper=5.0,value=0.77)
ignore_length = Uniform('ignore_length', lower=0.0, upper=5.0,value=0.250)
attract_length = Uniform('attract_length', lower=0.5, upper=20.0,value=3.175)
attract_exponent = Uniform('attract_exponent', lower=0.0, upper=5.0,value=2.36)
attract_angle = Uniform('attract_angle', lower=0, upper=pi,value=0.189)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.967)
alpha = Uniform('alpha',lower=0, upper=1,value=0.37)
beta = Uniform('beta',lower=0, upper=1,value=0.433)

neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')

# variable to normalize the move step lengths for the alignment rule
dists = neighbours[:,:,4]
stepLen=np.mean(dists[dists>0])
    
@deterministic(plot=False)
def social_vector(at_l=attract_length, at_a=attract_angle, at_de=attract_exponent, al_w=align_weight, ig=ignore_length):
    if at_l<0.5:
        at_l=0.5

    xj = (neighbours[:,:,0]*np.cos(neighbours[:,:,1]))+(np.cos(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])
    yj = (neighbours[:,:,0]*np.sin(neighbours[:,:,1]))+(np.sin(neighbours[:,:,3])*(al_w/stepLen)*neighbours[:,:,4])
        
    anglesj=np.arctan2(yj,xj)
    n_weights = np.exp(-(neighbours[:,:,0]/at_l)**at_de)
    n_weights[neighbours[:,:,0]<=ig]=0.0
    n_weights[(neighbours[:,:,1]<-at_a)|(neighbours[:,:,1]>at_a)]=0.0

    xsv = np.sum(np.cos(anglesj)*n_weights,1)
    ysv = np.sum(np.sin(anglesj)*n_weights,1)

    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out


@stochastic(observed=True)
def moves(social=rho_s, al=alpha, be=beta, sv=social_vector, value=mvector):
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
