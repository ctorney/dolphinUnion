
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

__all__ = ['rho_m','rho_e','beta','mvector']



rho_m = Uniform('rho_m',lower=0, upper=1,value=0.9187)
rho_e = Uniform('rho_e',lower=0, upper=1,value=0.9524)
beta = Uniform('beta',lower=0, upper=1,value=0.1342)


mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')
evector = evector[np.isfinite(mvector)]
mvector = mvector[np.isfinite(mvector)]
nonnan = np.ones_like(evector)    
nonnan[np.isnan(evector)]=0.0
evector[np.isnan(evector)]=0.0




@stochastic(observed=True)
def moves(rm=rho_m, re=rho_e, be=beta, value=mvector):
    
    bes = be*nonnan
    
    wce = (1/(2*pi)) * (1-np.power(re,2))/(1+np.power(re,2)-2*re*np.cos((evector-mvector).transpose())) # weighted wrapped cauchy
    wcm = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # weighted wrapped cauchy
    wcc = (bes*wce+(1.0-bes)*wcm)
    return np.sum(np.log(wcc))



