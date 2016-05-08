
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

__all__ = ['attact_length','attract_angle','align_length','align_angle','align_weight','rho_s','rho_m','rho_e','alpha','beta','mvector']


attract_length = Uniform('attract_length', lower=0.5, upper=20.0,value=8.4077668)
align_length = Uniform('align_length', lower=0.5, upper=20.0,value=2.7530568)
attract_angle = Uniform('attract_angle', lower=0, upper=pi,value=0.20212)
align_angle = Uniform('align_angle', lower=0, upper=pi,value=0.40682)
align_weight = Uniform('align_weight', lower=0.0, upper=2.0,value=0.68941107)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.967754)
alpha = Uniform('alpha',lower=0, upper=1,value=0.404474)

rho_m = 0.92#Uniform('rho_m',lower=0, upper=1,value=0.9181)
rho_e = 0.93#Uniform('rho_e',lower=0, upper=1,value=0.9178)
beta = 0.136#Uniform('beta',lower=0, upper=1,value=0.136)


neighbours = np.load('../pdata/neighbours.npy')
mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')
    
@deterministic(plot=False)
def social_vector(at_l=attract_length, at_a=attract_angle, al_l=align_length, al_a=align_angle,  al_w=align_weight):
        
    n_weights = np.ones_like(neighbours[:,:,0],dtype=np.float64)
    n_weights[neighbours[:,:,0]==0]=0.0
    n_weights[neighbours[:,:,0]>at_l]=0.0
    n_weights[(neighbours[:,:,1]<-at_a)|(neighbours[:,:,1]>at_a)]=0.0

    na_weights = al_w*np.ones_like(neighbours[:,:,0],dtype=np.float64)
    na_weights[neighbours[:,:,0]==0]=0.0
    na_weights[neighbours[:,:,0]>al_l]=0.0
    na_weights[(neighbours[:,:,1]<-al_a)|(neighbours[:,:,1]>al_a)]=0.0
    
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1) + np.sum(np.cos(neighbours[:,:,3])*na_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1) + np.sum(np.sin(neighbours[:,:,3])*na_weights,1)
    
   
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out


@stochastic(observed=True)
def moves(social=rho_s, al=alpha, sv=social_vector, value=mvector):
    rm=rho_m
    re=rho_e
    be=beta
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



