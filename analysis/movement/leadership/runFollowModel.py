#tester


from matplotlib import pylab as plt
import numpy as np

import followerModel
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(followerModel)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=20000, burn=10000, thin=10,verbose=0)
mcplot(M)

np.save('leader_weightLL.npy',M.trace('lead_weight')[:])
np.save('rho_eLL.npy',M.trace('rho_e')[:])
np.save('rho_mLL.npy',M.trace('rho_m')[:])
np.save('rho_sLL.npy',M.trace('rho_s')[:])
np.save('betaLL.npy',M.trace('beta')[:])
np.save('alphaLL.npy',M.trace('alpha')[:])
