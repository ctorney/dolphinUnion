
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

__all__ = ['attract_exponent','attract_length','attract_angle','align_exponent','align_length','align_angle','align_weight','rho_s','rho_m','rho_e','alpha','beta','mvector']


attract_length = Uniform('attract_length', lower=0.5, upper=20.0,value=5.1245)
align_length = Uniform('align_length', lower=0.5, upper=20.0,value=5.1245)
attract_exponent = Uniform('attract_exponent', lower=0.5, upper=50.0,value=0.9369)
align_exponent = Uniform('align_exponent', lower=0.5, upper=50.0,value=0.9369)
attract_angle = Uniform('attract_angle', lower=0, upper=pi,value=0.2)
align_angle = Uniform('align_angle', lower=0, upper=pi,value=0.4)
align_weight = Uniform('align_weight', lower=0.0, upper=2.0,value=0.7)

rho_s = Uniform('rho_s',lower=0, upper=1,value=0.9681)
rho_m = Uniform('rho_m',lower=0, upper=1,value=0.9212)
rho_e = Uniform('rho_e',lower=0, upper=1,value=0.9550)
alpha = Uniform('alpha',lower=0, upper=1,value=0.40)
beta = Uniform('beta',lower=0, upper=1,value=0.1359)

neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')

    
@deterministic(plot=False)
def social_vector(al_w=align_weight, at_l=attract_length, at_de=attract_exponent, at_a=attract_angle, al_l=align_length, al_de=align_exponent, al_a=align_angle):
    n_weights = np.exp(-(neighbours[:,:,0]/at_l)**at_de)
    n_weights[neighbours[:,:,0]==0]=0.0
    n_weights[(neighbours[:,:,1]<-at_a)|(neighbours[:,:,1]>at_a)]=0.0

    na_weights = al_w*np.exp(-(neighbours[:,:,0]/al_l)**al_de)
    na_weights[neighbours[:,:,0]==0]=0.0
    na_weights[(neighbours[:,:,1]<-al_a)|(neighbours[:,:,1]>al_a)]=0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1) + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1) + np.sum(np.sin(neighbours[:,:,3])*na_weights,1)

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

