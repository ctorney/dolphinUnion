import math
import pymc
import imp
from pymc import MCMC
import numpy as np
import matplotlib.pyplot as plt
import decayModelAlign
import fixedSocialCls


inds = np.unique(uid)
Ni = 3

iters = 50000
burns = 25000
thins = 1
Irhos = np.empty((Ni,int((iters-burns)/thins)))
Ialphas = np.empty((Ni,int((iters-burns)/thins)))
Iig = np.empty((Ni,int((iters-burns)/thins)))
Iil = np.empty((Ni,int((iters-burns)/thins)))
Iia = np.empty((Ni,int((iters-burns)/thins)))
Iiw = np.empty((Ni,int((iters-burns)/thins)))
Iae = np.empty((Ni,int((iters-burns)/thins)))
for i in range(Ni):

    print('-------------')
    print('Processing individual ' + str(i+1) + ' out of ' + str(Ni))
    print('-------------')
    np.save('terribleHackClass.npy',np.array([i]))
    imp.reload(fixedSocialCls)
    M = MCMC(fixedSocialCls)
    M.sample(iter=iters, burn=burns, thin=thins,verbose=0)
#    Irhos[i,:]=M.trace('rho_s')[:]
    Ialphas[i,:]=M.trace('alpha')[:]
#    Iig[i,:]=M.trace('ignore_length')[:]
#    Iil[i,:]=M.trace('attract_length')[:]
#    Iia[i,:]=M.trace('attract_angle')[:]
#    Iiw[i,:]=M.trace('align_weight')[:]
#    Iae[i,:]=M.trace('attract_exponent')[:]
    
#np.save('Irhos.npy',Irhos)
np.save('Ialphas.npy',Ialphas)
#np.save('Iig.npy',Iig)
#np.save('Iil.npy',Iil)
#np.save('Iia.npy',Iia)
#np.save('Iiw.npy',Iiw)
#np.save('Iae.npy',Iae)
#
#np.save('mc_data/cma_at_a.npy',M.trace('attract_angle')[:])
#np.save('mc_data/cma_at_l.npy',M.trace('attract_length')[:])
#np.save('mc_data/cma_aw.npy',M.trace('align_weight')[:])
#np.save('mc_data/cma_ig.npy',M.trace('ignore_length')[:])
