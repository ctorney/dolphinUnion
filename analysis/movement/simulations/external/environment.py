
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

__all__ = ['rho_m','rho_e','beta','mvector']



rho_m = Uniform('rho_m',lower=0, upper=1,value=0.921)
beta = Uniform('beta',lower=0, upper=1,value=0.135)


mvector = np.load('./pdata/mvector.npy')
evector = np.load('./pdata/evector.npy')
evector = evector[np.isfinite(mvector)]
mvector = mvector[np.isfinite(mvector)]




@stochastic(observed=True)
def moves(rm=rho_m, be=beta, value=mvector):
    
    xvals = (be*np.cos(evector)+(1.0-be))
    yvals = (be*np.sin(evector))

    allV = np.arctan2(yvals,xvals)
    wce = (1/(2*pi)) * (1-(rm*rm))/(1+(rm*rm)-2*rm*np.cos((allV-mvector).transpose()))

    wce = wce[wce>0]
    return np.sum(np.log(wce))
    



