
from math import pi
import numpy as np






mvector = np.load('../pdata/mvector.npy')
evector = np.load('../pdata/evector.npy')
evector = evector[np.isfinite(mvector)]
mvector = mvector[np.isfinite(mvector)]



def moves(rm, be):
    
    xvals = (be*np.cos(evector)+(1.0-be))
    yvals = (be*np.sin(evector))

    allV = np.arctan2(yvals,xvals)
    wce = (1/(2*pi)) * (1-(rm*rm))/(1+(rm*rm)-2*rm*np.cos((allV-mvector).transpose()))

    
    return (np.log(wce))
    



