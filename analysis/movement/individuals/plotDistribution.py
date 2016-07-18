import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
iAlpha = np.load('Ialphas.npy')
iRho = np.load('Irhos.npy')

ir2 = np.reshape(iRho,np.size(iRho))
plt.figure()
plt.hist(ir2,50,range=[0.8,1])
irm=np.mean(iRho,1)

for i,ind in enumerate(iRho):
    
    [counts,vals]=np.histogram(ind,100,normed=True)
    mxv = np.argmax(counts)
    irm[i]=vals[mxv]

plt.figure()
plt.hist(iam,100)


ia2 = np.reshape(iAlpha,np.size(iAlpha))
plt.figure()
plt.hist(ia2,100)

iam=np.mean(iAlpha,1)


for i,ind in enumerate(iAlpha):
    
    [counts,vals]=np.histogram(ind,100,normed=True)
    mxv = np.argmax(counts)
    iam[i]=vals[mxv]



plt.figure()    

for i,ind in enumerate(iRho):
    
    if irm[i]>0.95:
        [counts,vals]=np.histogram(ind,100,normed=True)#,range=[0.9,1.0])
        plt.plot(vals[:-1],counts)

plt.figure()    

for i,ind in enumerate(iRho):
    
    if irm[i]<0.95:
        [counts,vals]=np.histogram(ind,100,normed=True)
        plt.plot(vals[:-1],counts)
        



plt.figure()
lows = iRho[irm<0.95]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True,range=[0.9,1.0])

#plt.figure()
lows = iRho[irm>0.95]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True,range=[0.9,1.0])


plt.hist(iRho[84],100)
plt.hist(iRho[139],100)
plt.hist(iAlpha[139],100)
plt.hist(iAlpha[84],100)



plt.figure()    

for i,ind in enumerate(iAlpha):
    
    if iam[i]>0.25:
        [counts,vals]=np.histogram(ind,100,normed=True)#,range=[0.9,1.0])
        plt.plot(vals[:-1],counts)

plt.figure()    

for i,ind in enumerate(iAlpha):
    
    if iam[i]<0.25:
        [counts,vals]=np.histogram(ind,100,normed=True)
        plt.plot(vals[:-1],counts)
        

plt.figure()
lows = iAlpha[iam<0.25]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True,range=[0.0,1.0])

plt.figure()
lows = iAlpha[iam>0.25]
lows=np.reshape(lows,np.size(lows))
plt.hist(lows,100,normed=True,range=[0.0,1.0])