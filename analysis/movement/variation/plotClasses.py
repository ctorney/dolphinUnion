import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
iAlpha = np.load('Ialphas.npy')
iRho = np.load('Irhos.npy')




plt.figure()    

for i,ind in enumerate(iAlpha):
    
    
        #[counts,vals]=np.histogram(ind,100,normed=True)
        plt.hist(ind,100,range=[0,0.6],label=str(i))
plt.legend()
