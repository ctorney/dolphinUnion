import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
iAlpha = np.load('Ialphas.npy')
iRho = np.load('Irhos.npy')

ir2 = np.reshape(iRho,np.size(iRho))
plt.figure()
plt.hist(ir2,50,range=[0.8,1])
irm=np.median(iRho,1)

for i,ind in enumerate(iRho):
    
    [counts,vals]=np.histogram(ind,100,normed=True)
    mxv = np.argmax(counts)
    irm[i]=vals[mxv]

plt.figure()
plt.hist(irm,50)


ia2 = np.reshape(iAlpha,np.size(iAlpha))
#plt.figure()
#plt.hist(ia2,100)

iam=np.median(iAlpha,1)


for i,ind in enumerate(iRho):
    
    [counts,vals]=np.histogram(ind,100,normed=True)
    mxv = np.argmax(counts)
    iam[i]=vals[mxv]


plt.figure()    

for i,ind in enumerate(iRho):
    
    if iam[i]<0.97:
        [counts,vals]=np.histogram(ind,100,normed=True,range=[0.9,1.0])
        if np.max(counts)>900:
            break
        plt.plot(vals[:-1],counts)

plt.figure()    

for i,ind in enumerate(iRho):
    
    if iam[i]>0.90:
        [counts,vals]=np.histogram(ind,100,normed=True)
        plt.plot(vals[:-1],counts)
        



plt.figure()
lows = iRho[iam<0.90]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True)

plt.figure()
lows = iRho[iam>0.90]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True)
