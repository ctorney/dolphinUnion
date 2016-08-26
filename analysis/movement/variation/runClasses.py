import math
import pymc
import imp
from pymc import MCMC
import numpy as np
import matplotlib.pyplot as plt
import decayModelAlign
#import fixedSocialCls


Ni = 3

iters = 400000
burns = 50000
thins = 10
Irhos = np.empty((Ni,int((iters-burns)/thins)))
Ialphas = np.empty((Ni,int((iters-burns)/thins)))
Ibetas = np.empty((Ni,int((iters-burns)/thins)))
Iig = np.empty((Ni,int((iters-burns)/thins)))
Iil = np.empty((Ni,int((iters-burns)/thins)))
Iia = np.empty((Ni,int((iters-burns)/thins)))
Iiw = np.empty((Ni,int((iters-burns)/thins)))
Iae = np.empty((Ni,int((iters-burns)/thins)))
for i in range(0, 1):

    print('-------------')
    print('Processing class ' + str(i+1) + ' out of ' + str(Ni))
    print('-------------')
    np.save('terribleHackClass.npy',np.array([i]))
    imp.reload(decayModelAlign)
    M = MCMC(decayModelAlign)
    M.sample(iter=iters, burn=burns, thin=thins,verbose=0)
    Irhos[i,:]=M.trace('rho_s')[:]
    Ialphas[i,:]=M.trace('alpha')[:]
    Ibetas[i,:]=M.trace('beta')[:]
    Iig[i,:]=M.trace('ignore_length')[:]
    Iil[i,:]=M.trace('attract_length')[:]
    Iia[i,:]=M.trace('attract_angle')[:]
    Iiw[i,:]=M.trace('align_weight')[:]
    Iae[i,:]=M.trace('attract_exponent')[:]
    
np.save('Irhos0.npy',Irhos)
np.save('Ialphas0.npy',Ialphas)
np.save('Ibetas0.npy',Ibetas)
np.save('Iig0.npy',Iig)
np.save('Iil0.npy',Iil)
np.save('Iia0.npy',Iia)
np.save('Iiw0.npy',Iiw)
np.save('Iae0.npy',Iae)
#
#np.save('mc_data/cma_at_a.npy',M.trace('attract_angle')[:])
#np.save('mc_data/cma_at_l.npy',M.trace('attract_length')[:])
#np.save('mc_data/cma_aw.npy',M.trace('align_weight')[:])
#np.save('mc_data/cma_ig.npy',M.trace('ignore_length')[:])
