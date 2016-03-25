
import os
import csv
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand

import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['decay_distance','decay_network','interaction_distance','interaction_network','interaction_angle','rho_s','rho_m','rho_e','alpha','beta','mvector','social_vector','desired_vector']


interaction_network = Uniform('interaction_network', lower=0.5, upper=20.0,value=5.6549)
interaction_distance = Uniform('interaction_distance', lower=0.5, upper=20.0,value=5.6549)
#ignore_length = DiscreteUniform('ignore_length', lower=1, upper=3)#,value=1.0)
decay_distance = Uniform('decay_distance', lower=0.5, upper=10.0,value=0.9280)
decay_network = Uniform('decay_network', lower=0.5, upper=10.0,value=0.9280)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi,value=0.3327)
rho_s = Uniform('rho_s',lower=0, upper=1,value=0.9524)
rho_m = Uniform('rho_m',lower=0, upper=1,value=0.9236)
rho_e = Uniform('rho_e',lower=0, upper=1,value=0.9554)
alpha = Uniform('alpha',lower=0, upper=1,value=0.3104)
beta = Uniform('beta',lower=0, upper=1,value=0.1569)
# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)

neighbours = np.load('../neighbours.npy')
mvector = np.load('../mvector.npy')
evector = np.load('../evector.npy')
sin_ev = np.sin(evector)
cos_ev = np.cos(evector)
    
@deterministic(plot=False)
def social_vector(iln=interaction_network, ild=interaction_distance, ded=decay_distance, den=decay_network, ia=interaction_angle):
        
    distances = neighbours[:,:,0]
    distances[(neighbours[:,:,0]==0)]=9999.0
    networkDist = np.argsort(distances,axis=1).astype(np.float32)
    networkDist = np.argsort(networkDist,axis=1).astype(np.float32)

    n_weights = np.exp(-((networkDist/iln)**den))*((neighbours[:,:,0]/ild)*np.exp((1.0/ded)*(1.0-(neighbours[:,:,0]/ild)**ded)))#*(0.5+0.5*np.tanh(ad*(ia-np.abs(neighbours[:,:,1]))))
    n_weights[(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
    n_weights[(neighbours[:,:,0]==0)]=0.0
 
    xsv = np.sum(np.cos(neighbours[:,:,1])*n_weights,1)
    ysv = np.sum(np.sin(neighbours[:,:,1])*n_weights,1)
    
    lens = np.sqrt(xsv**2+ysv**2)
    ysv[lens>1]=ysv[lens>1]/lens[lens>1]
    xsv[lens>1]=xsv[lens>1]/lens[lens>1]
    out = np.empty((len(mvector),2))

    out[:,0] = xsv
    out[:,1] = ysv
    
    return out


@stochastic(observed=True)
def moves(social=rho_s, rm=rho_m,re=rho_e,al=alpha, be=beta, sv=social_vector, value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    svv = np.arctan2(sv[:,1],sv[:,0])
    lens = np.sqrt(sv[:,1]**2+sv[:,0]**2)
    als = al*lens
    socials=lens*social
    wcs = (1/(2*pi)) * (1-np.power(socials,2))/(1+np.power(socials,2)-2*socials*np.cos((svv-mvector).transpose())) # weighted wrapped cauchy
    wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = als*wcs + (1.0-als)*(be*wce+(1.0-be)*wcm)
    return np.sum(np.log(wcc))




#
#@stochastic(observed=True)
#def moves(il=interaction_length,ig=ignore_length, ia=interaction_angle, social=rho, al=alpha, be=beta,value=mvector):
#    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
#    # and the assumed interaction function
#    
#    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
#    #lambdas[np.abs(mvector)>pi]=pi
#    #dv[np.abs(mvector)>(1-al)*pi]=pi
#    #dv[np.abs(mvector)<(1-al)*pi]=mvector[np.abs(mvector)<(1-al)*pi]/(1-al)
# #   dv =np.arctan2( (np.sin(mvector)-(1.0-beta)*(1.0-alpha)*np.sin(evector)), (np.cos(mvector)-(1.0-beta)*(1.0-alpha)*np.cos(evector)))
#   # first calculate all the rhos
#    n_weights = np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
#    n_weights[(neighbours[:,:,0]==0)|(neighbours[:,:,1]<-ia)|(neighbours[:,:,1]>ia)]=0.0
#    #n_weights = np.exp(-np.abs(neighbours[:,:,1])/ia)*np.exp(-neighbours[:,:,0]/il)*np.tanh(neighbours[:,:,0]/ig)
#    #n_weights[(neighbours[:,:,0]==0)]=0.0
#    
#    xpos = np.cos(neighbours[:,:,1])*n_weights
#    ypos = np.sin(neighbours[:,:,1])*n_weights
#    
#    sv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    ysv = np.sin(sv)
#    xsv = np.cos(sv)
#    xsv[(np.sum(ypos,1)==0)&(np.sum(xpos,1)==0)] = 0.0
#    ally = be*ysv+(1.0-be)*(1.0-al)*np.sin(evector)
#    allx = be*xsv+(1.0-be)*(al*np.ones_like(mvector)+(1.0-al)*np.cos(evector))
#    #dv = np.arctan2(np.sum(ypos,1), np.sum(xpos,1))
#    dv = np.arctan2(ally,allx)
#    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
#    #nc = np.sum(np.abs(rhos),1) # normalizing constant
#
#    #wwc = ((rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((dv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
#    wcs = (1/(2*pi)) * (1-np.power(social,2))/(1+np.power(social,2)-2*social*np.cos((dv-mvector).transpose())) # weighted wrapped cauchy
# #   wcs[(np.sum(ypos,1)==0)&(np.sum(xpos,1)==0)] = 1/(2*pi)
# #   wce = (1/(2*pi)) * (1-np.power(be,2))/(1+np.power(be,2)-2*be*np.cos((dv-evector).transpose())) # weighted wrapped cauchy
#    # sum along the individual axis to get the total compound cauchy
#    #wwc = np.sum(wwc,1)/nc
#    #wwc[np.isnan(wwc)]=1/(2*pi)
#    
#  #  wwc = (social/(be+social))*wcs + (be/(be+social))*wce
##    # next we want to split desired vector into social and environmental vector
##    sv = np.zeros_like(mvector)
##    # desired vector  = b*env_vector + (1-b)*social_vector
##    sv = (dv - be*evector)/(1-be)
##    sv[np.isnan(sv)]=pi
##    sv[np.abs(sv)>pi]=pi
##    
##    # first calculate all the rhos
##    rhos = np.zeros_like(neighbours[:,:,0])
##    rhos[(neighbours[:,:,0]>0)&(neighbours[:,:,0]<il)&(neighbours[:,:,1]>-ia)&(neighbours[:,:,1]<ia)]=social
##    
##    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
##    nc = np.sum(np.abs(rhos),1) # normalizing constant
##
##    wwc = (np.abs(rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((sv-neighbours[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
##    # sum along the individual axis to get the total compound cauchy
##    wwc = np.sum(wwc,1)/nc
##    wwc[np.isnan(wwc)]=1/(2*pi)
#    return np.sum(np.log(wcs))
