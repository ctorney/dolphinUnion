
import os
import math

from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
from math import pi
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['rho_m','mvector']


mvector = np.load('../pdata/mvector.npy')
mvector = mvector[np.isfinite(mvector)]


def moves(rm):
    wcc = (1/(2*pi)) * (1-np.power(rm,2))/(1+np.power(rm,2)-2*rm*np.cos((-mvector).transpose())) # wrapped cauchy
    return (np.log(wcc))



