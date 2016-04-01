#tester


from matplotlib import pylab as plt
import numpy as np

import followerModel
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(followerModel)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=200000, burn=10000, thin=10,verbose=0)
mcplot(M)

np.save('Frho_e.npy',M.trace('rho_e')[:])
np.save('Frho_m.npy',M.trace('rho_m')[:])
np.save('Frho_s.npy',M.trace('rho_s')[:])
np.save('Fbeta.npy',M.trace('beta')[:])
np.save('Falpha.npy',M.trace('alpha')[:])
np.save('Frho_eL.npy',M.trace('rho_eL')[:])
np.save('Frho_mL.npy',M.trace('rho_mL')[:])
np.save('Frho_sL.npy',M.trace('rho_sL')[:])
np.save('FbetaL.npy',M.trace('betaL')[:])
np.save('FalphaL.npy',M.trace('alphaL')[:])
np.save('Flead_weight.npy',M.trace('lead_weight')[:])

np.save('Finteraction_angle.npy',M.trace('interaction_angle')[:])
np.save('Finteraction_length.npy',M.trace('interaction_length')[:])


