import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
iAlpha = np.load('Ialphas.npy')
iBeta = np.load('Ibetas.npy')
iRho = np.load('Irhos.npy')
iAngle0 = np.load('Iiw0.npy')
iAngle = np.load('Iiw.npy')

thisHist =iAngle# (1.0-iAlpha)*(1.0-iBeta)


plt.figure()    
#plt.hist(iAngle0[0],100,range=[0.0,3.0],label='adults')
#plt.hist(iAngle[2],100,range=[0.0,3.0],label='calves')

#for i,ind in enumerate(thisHist):
    
    
        #[counts,vals]=np.histogram(ind,100,normed=True)
        #plt.hist(ind,100,range=[0.0,3.0],label=str(i))
        
        
plt.hist(iAngle0[0],100,range=[0.0,1.5],label='adults',normed=True,histtype='stepfilled',color='midnightblue',alpha=0.9)
plt.hist(iAngle[2],100,range=[0.0,1.5],label='calves',normed=True,histtype='stepfilled',color='firebrick',alpha=0.9)


plt.legend()

plt.ylabel('posterior probability density',fontsize=16)
plt.xlabel('alignment',fontsize=16)
plt.xlim([0,1.5])
plt.savefig("align.tiff",bbox_inches='tight',dpi=300)
plt.savefig("align.png",bbox_inches='tight',dpi=300)

