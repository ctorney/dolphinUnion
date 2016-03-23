#tester


from matplotlib import pylab as plt
import numpy as np

import leaderModel
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(leaderModel)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=20000, burn=1000, thin=10,verbose=0)
mcplot(M)
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
aa = (M.trace('alpha')[:])
bb = (M.trace('beta')[:])
mv=(1-bb)*(1-aa)
sv=aa
ev=(bb)*(1-aa)
plt.hist(mv,normed=True, label='memory')
plt.hist(sv,normed=True, label='social')
plt.hist(ev,normed=True, label='environment')

plt.legend(loc='upper center')
plt.xlim(0,1)
plt.show()
plt.savefig('heading_weights.png')


plt.hist((M.trace('rho_s')[:]),normed=True, label='s')
plt.hist((M.trace('rho_m')[:]),normed=True, label='m')
plt.hist((M.trace('rho_e')[:]),normed=True, label='environment')

plt.xlim(0.7,1.0)
plt.legend(loc='upper center')
plt.show()


plt.hist((M.trace('beta')[:]),normed=True,label='beta')

plt.xlim(0.9,0.94)
plt.legend(loc='upper center')
plt.show()
plt.savefig('rho.png')


mrho = 0.96
xx = np.linspace(-3.142,3.142,1000)
yy = (1.0/(2.0*math.pi))*(1.0-mrho**2)/(1.0+mrho**2-2*mrho*np.cos(xx))
plt.xlim(-3.142,3.142)
plt.ylim(0,4)
plt.plot(xx,yy,color='blue',linewidth=2)
plt.fill_between(xx, 0, yy, color='blue', alpha=.25)

np.save('rho_e.npy',M.trace('rho_e')[:])
np.save('rho_m.npy',M.trace('rho_m')[:])
np.save('rho_s.npy',M.trace('rho_s')[:])
np.save('interaction_angle.npy',M.trace('interaction_angle')[:])
np.save('angle_decay.npy',M.trace('angle_decay')[:])
np.save('beta.npy',M.trace('beta')[:])
np.save('interaction_length.npy',M.trace('interaction_length')[:])
np.save('decay_exponent.npy',M.trace('decay_exponent')[:])
np.save('alpha.npy',M.trace('alpha')[:])



