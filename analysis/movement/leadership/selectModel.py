#tester


from matplotlib import pylab as plt
import numpy as np


import constantModel
import leaderModel
import followerModel
import pymc
from pymc import MAP
from pymc.Matplot import plot as mcplot


CM = MAP(constantModel)
CM.fit(method ='fmin', iterlim=100000, tol=.000001)
print(CM.AIC)
print(CM.BIC)


LM = MAP(leaderModel)
LM.fit(method ='fmin', iterlim=100000, tol=.000001)
print(LM.AIC)
print(LM.BIC)

FM = MAP(followerModel)
FM.fit(method ='fmin', iterlim=100000, tol=.000001)
print(FM.AIC)
print(FM.BIC)