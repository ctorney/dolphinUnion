
import os
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand

from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['rho_m','mvector']

rho_m = Uniform('rho_m',lower=0, upper=1,value=0.5)
mvector = np.load('../pdata/mvector.npy')

@stochastic(observed=True)
def moves(rm=rho_m, value=mvector):
    wcc = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # wrapped cauchy
    wcc= wcc[wcc>0]
    return np.sum(np.log(wcc))



