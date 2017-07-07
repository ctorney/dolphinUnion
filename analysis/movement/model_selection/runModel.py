
from matplotlib import pylab as plt
import numpy as np

import corRandomWalk
import environment
import constantModel
import constantModelAlign
import decayModel
import decayModelAlign
import networkModelMC
import networkModelAlignMC


import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot


# random walk model
M = MCMC(corRandomWalk)
M.sample(iter=4000, burn=50, thin=10,verbose=0)
#np.save('mc_data/crw_rho_m.npy',M.trace('rho_m')[:])

# environment model
#M = MCMC(environment)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/env_rho_m.npy',M.trace('rho_m')[:])
#np.save('mc_data/env_beta.npy',M.trace('beta')[:])

## constant model
#M = MCMC(constantModel)
#M.use_step_method(pymc.AdaptiveMetropolis, [constantModel.interaction_angle,constantModel.interaction_length,constantModel.ignore_length], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/cm_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/cm_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/cm_beta.npy',M.trace('beta')[:])
#
#np.save('mc_data/cm_ia.npy',M.trace('interaction_angle')[:])
#np.save('mc_data/cm_il.npy',M.trace('interaction_length')[:])
#np.save('mc_data/cm_ig.npy',M.trace('ignore_length')[:])
#
## constant model align
#M = MCMC(constantModelAlign)
#M.use_step_method(pymc.AdaptiveMetropolis, [constantModelAlign.attract_angle,constantModelAlign.align_weight,constantModelAlign.attract_length,constantModelAlign.ignore_length], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/cma_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/cma_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/cma_beta.npy',M.trace('beta')[:])
#
#np.save('mc_data/cma_at_a.npy',M.trace('attract_angle')[:])
#np.save('mc_data/cma_at_l.npy',M.trace('attract_length')[:])
#np.save('mc_data/cma_aw.npy',M.trace('align_weight')[:])
#np.save('mc_data/cma_ig.npy',M.trace('ignore_length')[:])

## decay model
#M = MCMC(decayModel)
#M.use_step_method(pymc.AdaptiveMetropolis, [decayModel.interaction_angle,decayModel.interaction_length,decayModel.ignore_length,decayModel.decay_exponent], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/dm_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/dm_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/dm_beta.npy',M.trace('beta')[:])

#np.save('mc_data/dm_ia.npy',M.trace('interaction_angle')[:])
#np.save('mc_data/dm_il.npy',M.trace('interaction_length')[:])
#np.save('mc_data/dm_de.npy',M.trace('decay_exponent')[:])
#np.save('mc_data/dm_ig.npy',M.trace('ignore_length')[:])

## decay model align
#M = MCMC(decayModelAlign)
#M.use_step_method(pymc.AdaptiveMetropolis, [decayModelAlign.attract_exponent,decayModelAlign.attract_angle,decayModelAlign.align_weight,decayModelAlign.attract_length,decayModelAlign.ignore_length], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/dma_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/dma_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/dma_beta.npy',M.trace('beta')[:])

#np.save('mc_data/dma_at_a.npy',M.trace('attract_angle')[:])
#np.save('mc_data/dma_at_l.npy',M.trace('attract_length')[:])
#np.save('mc_data/dma_aw.npy',M.trace('align_weight')[:])
#np.save('mc_data/dma_at_e.npy',M.trace('attract_exponent')[:])
#np.save('mc_data/dma_ig.npy',M.trace('ignore_length')[:])


# network model
#M = MCMC(networkModelMC)
#M.use_step_method(pymc.AdaptiveMetropolis, [networkModelMC.interaction_angle,networkModelMC.interaction_length,networkModelMC.ignore_length], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/nm_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/nm_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/nm_beta.npy',M.trace('beta')[:])

#np.save('mc_data/nm_ia.npy',M.trace('interaction_angle')[:])
#np.save('mc_data/nm_il.npy',M.trace('interaction_length')[:])
#np.save('mc_data/nm_ig.npy',M.trace('ignore_length')[:])

# network model with alignment
#M = MCMC(networkModelAlignMC)
#M.use_step_method(pymc.AdaptiveMetropolis, [networkModelAlignMC.interaction_angle,networkModelAlignMC.interaction_length,networkModelAlignMC.align_weight,networkModelAlignMC.ignore_length], delay=1000)
#M.sample(iter=400000, burn=50000, thin=10,verbose=0)
#np.save('mc_data/nma_rho_s.npy',M.trace('rho_s')[:])
#np.save('mc_data/nma_alpha.npy',M.trace('alpha')[:])
#np.save('mc_data/nma_beta.npy',M.trace('beta')[:])
#
#np.save('mc_data/nma_ia.npy',M.trace('interaction_angle')[:])
#np.save('mc_data/nma_il.npy',M.trace('interaction_length')[:])
#np.save('mc_data/nma_aw.npy',M.trace('align_weight')[:])
#np.save('mc_data/nma_ig.npy',M.trace('ignore_length')[:])
