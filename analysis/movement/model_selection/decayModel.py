
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

__all__ = ['ignore_length','decay_exponent','interaction_length','interaction_angle','rho_s','rho_m','rho_e','alpha','beta','mvector']


ignore_length = Uniform('ignore_length', lower=0.0, upper=5.0,value=0.81)
interaction_length = Uniform('interaction_length', lower=0.5, upper=20.0,value=2.89)
decay_exponent = Uniform('decay_exponent', lower=0.0, upper=3.0,value=1.03)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi,value=0.19)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.967)
alpha = Uniform('alpha',lower=0, upper=1,value=0.316)

rho_m = 0.937
rho_e = 0.956
beta = 0.126

neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')

    
@deterministic(plot=False)
def social_vector(il=interaction_length, de=decay_exponent, ia=interaction_angle, ig=ignore_length):
    n_weights = np.exp(-(neighbours[:,:,0]/il)**de)

    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[neighbours[:,:,0]<=ig]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out


@stochastic(observed=True)
def moves(social=rho_s,al=alpha,sv=social_vector, value=mvector):
    rm=rho_m
    re=rho_e
    be=beta
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    als = al*np.ones_like(svv)
    als[(sv[:,1]==0)&(sv[:,0]==0)]=0
    
    wcs = (1/(2*pi)) * (1-(social*social))/(1+(social*social)-2*social*np.cos((svv-mvector).transpose()))
    wce = (1/(2*pi)) * (1-(re*re))/(1+(re*re)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-(rm*rm))/(1+(rm*rm)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy

    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    wcc = wcc[wcc>0]
    return np.sum(np.log(wcc))

