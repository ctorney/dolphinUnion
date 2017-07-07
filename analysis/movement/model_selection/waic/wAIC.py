
from matplotlib import pylab as plt
import numpy as np


#import environment
#import constantModel
#import constantModelAlign
#import decayModel
#import decayModelAlign
#import networkModelMC
#import networkModelAlignMC


#%%
# random walk model
import corRandomWalk
rho_m = np.load('../mc_data/crw_rho_m.npy')
rho_m=rho_m[::10]

mvector = np.load('../../pdata/mvector.npy')
mvector = mvector[np.isfinite(mvector)]

mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=corRandomWalk.moves(rho_m[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)
lphat = np.sum(corRandomWalk.moves(np.mean(rho_m)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("CRW ", waic, p_waic, dic)
#%%
# environment model
import environment
skip=10
rho_m = np.load('../mc_data/env_rho_m.npy')
beta = np.load('../mc_data/env_beta.npy')
rho_m=rho_m[::skip]
beta=beta[::skip]

mvector = np.load('../../pdata/mvector.npy')
mvector = mvector[np.isfinite(mvector)]

mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=environment.moves(rho_m[i],beta[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)
lphat = np.sum(environment.moves(np.mean(rho_m),np.mean(beta)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))


vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("env ", waic,p_waic,dic)

#%%
# constant model align
import constantModel

skip=10
rho_m=np.load('../mc_data/cm_rho_s.npy')
alpha=np.load('../mc_data/cm_alpha.npy')
beta=np.load('../mc_data/cm_beta.npy')
ia = np.load('../mc_data/cm_ia.npy')
il = np.load('../mc_data/cm_il.npy')
ig = np.load('../mc_data/cm_ig.npy')

rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]


mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=constantModel.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)
lphat = np.sum(constantModel.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(ig)))

epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("cm ", waic,p_waic,dic)

#%%
# constant model align
impor constantModelAlign

skip=10
rho_m=np.load('../mc_data/cma_rho_s.npy')
alpha=np.load('../mc_data/cma_alpha.npy')
beta=np.load('../mc_data/cma_beta.npy')
ia = np.load('../mc_data/cma_at_a.npy')
il = np.load('../mc_data/cma_at_l.npy')
ig = np.load('../mc_data/cma_ig.npy')
aw = np.load('../mc_data/cma_aw.npy')
rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]
aw=aw[::skip]


mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=constantModelAlign.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],aw[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)

lphat = np.sum(constantModelAlign.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(aw),np.mean(ig)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("cma ", waic,p_waic,dic)


#%%
# decay model 
import decayModel

skip=10
rho_m=np.load('../mc_data/dm_rho_s.npy')
alpha=np.load('../mc_data/dm_alpha.npy')
beta=np.load('../mc_data/dm_beta.npy')
ia = np.load('../mc_data/dm_ia.npy')
il = np.load('../mc_data/dm_il.npy')
de = np.load('../mc_data/dm_de.npy')
ig = np.load('../mc_data/dm_ig.npy')

rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]
de=de[::skip]



mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=decayModel.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],de[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)
lphat = np.sum(decayModel.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(de),np.mean(ig)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("dm ", waic, p_waic,dic)

#%%
# decay model align
import decayModelAlign

skip=10
rho_m=np.load('../mc_data/dma_rho_s.npy')
alpha=np.load('../mc_data/dma_alpha.npy')
beta=np.load('../mc_data/dma_beta.npy')
ia = np.load('../mc_data/dma_at_a.npy')
il = np.load('../mc_data/dma_at_l.npy')
de = np.load('../mc_data/dma_at_e.npy')
ig = np.load('../mc_data/dma_ig.npy')
aw = np.load('../mc_data/dma_aw.npy')
rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]
de=de[::skip]
aw=aw[::skip]


mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=decayModelAlign.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],de[i],aw[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)

lphat = np.sum(decayModelAlign.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(de),np.mean(aw),np.mean(ig)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("dma ", waic, p_waic,dic)

#%%
# network model 
import networkModel

skip=10
rho_m=np.load('../mc_data/nm_rho_s.npy')
alpha=np.load('../mc_data/nm_alpha.npy')
beta=np.load('../mc_data/nm_beta.npy')
ia = np.load('../mc_data/nm_ia.npy')
il = np.load('../mc_data/nm_il.npy')
ig = np.load('../mc_data/nm_ig.npy')

rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]




mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=networkModel.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)

lphat = np.sum(networkModel.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(ig)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("nm ", waic, p_waic,dic)

#%%
# decay model align
import networkModelAlign

skip=10
rho_m=np.load('../mc_data/nma_rho_s.npy')
alpha=np.load('../mc_data/nma_alpha.npy')
beta=np.load('../mc_data/nma_beta.npy')
ia = np.load('../mc_data/nma_ia.npy')
il = np.load('../mc_data/nma_il.npy')
ig = np.load('../mc_data/nma_ig.npy')
aw = np.load('../mc_data/nma_aw.npy')
rho_m=rho_m[::skip]
alpha=alpha[::skip]
beta=beta[::skip]
ia=ia[::skip]
il=il[::skip]
ig=ig[::skip]
aw=aw[::skip]


mvector = np.load('../../pdata/mvector.npy')


mvlen = len(mvector)
del mvector
log_py=np.zeros((len(rho_m),mvlen))

for i in range(len(rho_m)):
    log_py[i,:]=networkModelAlign.moves(rho_m[i],alpha[i],beta[i],il[i],ia[i],aw[i],ig[i])

lppd_i = np.log(np.mean(np.exp(log_py), axis=0))

lppd = np.sum(lppd_i)
lphat = np.sum(networkModelAlign.moves(np.mean(rho_m),np.mean(alpha),np.mean(beta),np.mean(il),np.mean(ia),np.mean(aw),np.mean(ig)))
epost=np.mean(np.sum(log_py,axis=1))
dic = -2*lphat + 2*(2*(lphat - epost))

vars_lpd = np.var(log_py, axis=0)

waic_i = - 2 * (lppd_i - vars_lpd)

waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

waic = np.sum(waic_i)

p_waic = np.sum(vars_lpd)

print("nma ", waic, p_waic, dic)

