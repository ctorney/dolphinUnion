#tester


from matplotlib import pylab as plt
import numpy as np
from sklearn.neighbors import KernelDensity

import constantModel
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot


#np.save('rho_e.npy',M.trace('rho_e')[:])
#np.save('rho_m.npy',M.trace('rho_m')[:])
rhos = np.load('rho_s.npy')
rhosL = np.load('rho_sL.npy')

bb = np.load('beta.npy')
aa = np.load('alpha.npy')


bbL = np.load('betaL.npy')
aaL = np.load('alphaL.npy')
plt.figure()
X_plot = np.linspace(0.92, 1, 1000)[:, np.newaxis]

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(rhos[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'-',linewidth=2,label='followers')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4)#, fc='#AAAAFF')
kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(rhosL[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'-',linewidth=2,label='leaders')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='green')#, fc='#AAAAFF')
plt.savefig('lf_social_strength.png')
plt.legend(loc='upper center')
plt.figure()


mv=(1-bb)*(1-aa)
sv=aa
ev=(bb)*(1-aa)

X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(mv[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'k-',linewidth=2,label='persistence')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='k')#, fc='#AAAAFF')

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(sv[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'-',linewidth=2,label='social')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4)#, fc='#AAAAFF')

kde = KernelDensity(kernel='gaussian', bandwidth=0.0001).fit(ev[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'-',linewidth=2,label='environment')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='green')#, fc='#AAAAFF')

mv=(1-bbL)*(1-aaL)
sv=aaL
ev=(bbL)*(1-aaL)

X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(mv[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'k--',linewidth=2,label='persistence leader')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='k')#, fc='#AAAAFF')

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(sv[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'--',linewidth=2,label='social leader',color='red')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='red')#, fc='#AAAAFF')

kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(ev[:,np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens),'--',linewidth=2,label='environment leader',color='green')
plt.fill_between(X_plot[:, 0],np.zeros_like(X_plot[:, 0]), np.exp(log_dens),alpha=0.4,color='green')#, fc='#AAAAFF')

plt.legend(loc='upper center')
plt.xlabel('weighting')
plt.ylabel('likelihood')
plt.savefig('lf_weights.png')

#

#from pylab import hist, show

#
#plt.hist(M.trace('reprho')[:])
#plt.xlim(0,1)

#plt.title('repulsion strength')

#plt.savefig('repulsion_strength.png')
#plt.show()
#plt.hist(M.trace('attrho')[:])
#plt.xlim(0,1)

#plt.title('attraction strength')

#plt.savefig('attraction_strength.png')
#plt.show()
#plt.hist(M.trace('replen')[:])
#plt.xlim(0,5)
#plt.title('repulsion length')



#plt.savefig('repulsion_length.png')
#plt.show()
#plt.hist(M.trace('eta')[:])
#plt.xlim(0,1)

#plt.title('autocorrelation')

#plt.savefig('autocorrelation.png')
#plt.show()

#show()
#aa = (M.trace('alpha')[:])
#bb = (M.trace('beta')[:])
#mv=(1-bb)*(1-aa)
#sv=aa
#ev=(bb)*(1-aa)
#plt.hist(mv,normed=True, label='memory')
#plt.hist(sv,normed=True, label='social')
#plt.hist(ev,normed=True, label='environment')
#
#plt.legend(loc='upper center')
#plt.xlim(0,1)
#plt.show()
#plt.savefig('heading_weights.png')
#
#
#plt.hist((M.trace('rho_s')[:]),normed=True, label='s')
#plt.hist((M.trace('rho_m')[:]),normed=True, label='m')
#plt.hist((M.trace('rho_e')[:]),normed=True, label='environment')
#
#plt.xlim(0.7,1.0)
#plt.legend(loc='upper center')
#plt.show()
#
#
#plt.hist((M.trace('beta')[:]),normed=True,label='beta')
#
#plt.xlim(0.9,0.94)
#plt.legend(loc='upper center')
#plt.show()
#plt.savefig('rho.png')
#
#
#mrho = 0.96
#xx = np.linspace(-3.142,3.142,1000)
#yy = (1.0/(2.0*math.pi))*(1.0-mrho**2)/(1.0+mrho**2-2*mrho*np.cos(xx))
#plt.xlim(-3.142,3.142)
#plt.ylim(0,4)
#plt.plot(xx,yy,color='blue',linewidth=2)
#plt.fill_between(xx, 0, yy, color='blue', alpha=.25)
#
#
#
