
import os
import csv
import math
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['interaction_length','interaction_angle','rho','alpha','beta','mvector']


interaction_length = Uniform('interaction_length', lower=0.5, upper=20.0)
interaction_angle = Uniform('interaction_angle', lower=0, upper=pi)
rho = Uniform('rho',lower=0, upper=1)
alpha = Uniform('alpha',lower=0, upper=1)
beta = Uniform('beta',lower=0, upper=1)

# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)

neighbours = np.load('neighbours.npy')
mvector = np.load('mvector.npy')
evector = np.load('evector.npy')
    


@stochastic(observed=True)
def moves(il=interaction_length,ia=interaction_angle, social=rho, al=alpha, be=beta,value=mvector):
    # this is the main function that calculates the log probability of all the moves based on the parameters that are passed in
    # and the assumed interaction function
    
    dv = np.zeros_like(mvector) # these are the headings (desired vector) without the autocorrelation; new heading = (eta)*(old heading) + (1-eta)*dv
    #lambdas[np.abs(mvector)>pi]=pi
    dv[np.abs(mvector)>(1-al)*pi]=pi
    dv[np.abs(mvector)<(1-al)*pi]=mvector[np.abs(mvector)<(1-al)*pi]/(1-al)
    
    # next we want to split desired vector into social and environmental vector
    sv = np.zeros_like(mvector)
    # desired vector  = b*env_vector + (1-b)*social_vector
    sv = (dv - be*ev)/(1-ev)
    sv[np.isnan(sv)]=pi
    sv[np.abs(sv)>pi]=pi
    
    # first calculate all the rhos
    rhos = np.zeros_like(neighbours[:,:,0])
    rhos[(neighbours[:,:,0]>0)&(neighbours[:,:,0]<il)&(neighbours[:,:,1]>-ia)&(neighbours[:,:,1]<ia)]=social
    
    # this isn't necessary here but if there are larger groups each neighbour has to be included and the total normalized
    nc = np.sum(np.abs(rhos),1) # normalizing constant

    wwc = (np.abs(rhos))*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((sv-dparams[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    # sum along the individual axis to get the total compound cauchy
    wwc = np.sum(wwc,1)/nc
    wwc[np.isinf(wwc)]=1/(2*pi)
    return np.sum(np.log(wwc))
