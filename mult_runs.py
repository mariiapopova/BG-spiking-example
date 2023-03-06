# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:09:29 2021

@author: Maria
"""
from initial import Initial
import matplotlib.pyplot as plt
from findfreq import *
from calculateEI import *
import datetime
import pickle
import numpy as np

n=100 #number of runs
 
name = "Data/data_%s.pckl" % datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
file = open(name, 'wb')

for i in range(n):
    
    vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx,\
    vsn1, vge1, vgi1, vppn1, vth1, vstr1, vsnr1, vprf1, vcnf1, vlc1, vsnc1, vctx1,\
    vsn2, vge2, vgi2, vppn2, vth2, vstr2, vsnr2, vprf2, vcnf2, vlc2, vsnc2, vctx2,\
    t, timespike, tmax=Initial()
    
    #for healthy
    ##calculate freqeuncy for plotting
    fr1=findfreq(vsn[0,:])
    fr2=findfreq(vge[0,:])
    fr3=findfreq(vgi[0,:])  
    fr4=findfreq(vppn[0,:])
    fr5=findfreq(vth[0,:])
    fr6=findfreq(vstr[0,:])
    fr7=findfreq(vsnr[0,:])
    fr8=findfreq(vprf[0,:])
    fr9=findfreq(vcnf[0,:])
    fr10=findfreq(vlc[0,:])
    fr11=findfreq(vsnc[0,:])
    fr12=findfreq(vctx[0,:])

    ##Calculation of error index
    GN=calculateEI(t,vth,timespike,tmax) #for thalamus
    #GN=calculateEI(t,vppn,timespike,tmax) #for PPN
    
    
    #for pd
    ##calculate freqeuncy for plotting
    fr1_1=findfreq(vsn1[0,:])
    fr2_1=findfreq(vge1[0,:])
    fr3_1=findfreq(vgi1[0,:])  
    fr4_1=findfreq(vppn1[0,:])
    fr5_1=findfreq(vth1[0,:])
    fr6_1=findfreq(vstr1[0,:])
    fr7_1=findfreq(vsnr1[0,:])
    fr8_1=findfreq(vprf1[0,:])
    fr9_1=findfreq(vcnf1[0,:])
    fr10_1=findfreq(vlc1[0,:])
    fr11_1=findfreq(vsnc1[0,:])
    fr12_1=findfreq(vctx1[0,:])
    
    ##Calculation of error index
    GN1=calculateEI(t,vth1,timespike,tmax) #for thalamus
    #GN1=calculateEI(t,vppn1,timespike,tmax) #for PPN
    
    
    #for dbs
    ##calculate freqeuncy for plotting
    fr1_2=findfreq(vsn2[0,:])
    fr2_2=findfreq(vge2[0,:])
    fr3_2=findfreq(vgi2[0,:])  
    fr4_2=findfreq(vppn2[0,:])
    fr5_2=findfreq(vth2[0,:])
    fr6_2=findfreq(vstr2[0,:])
    fr7_2=findfreq(vsnr2[0,:])
    fr8_2=findfreq(vprf2[0,:])
    fr9_2=findfreq(vcnf2[0,:])
    fr10_2=findfreq(vlc2[0,:])
    fr11_2=findfreq(vsnc2[0,:])
    fr12_2=findfreq(vctx2[0,:])
    
    ##Calculation of error index
    GN2=calculateEI(t,vth2,timespike,tmax) #for thalamus
    #GN2=calculateEI(t,vppn,timespike,tmax) #for PPN
    
    ##create dictionaries
    h_data = {"sn": fr1, "ge": fr2, 'gi': fr3, 'ppn': fr4, 'th': fr5, 'str': fr6, 'snr': fr7, 'prf': fr8, 'cnf': fr9,\
              'lc': fr10, 'snc': fr11, 'ctx': fr12, 'ei': GN}    

    pd_data = {"sn": fr1_1, "ge": fr2_1, 'gi': fr3_1, 'ppn': fr4_1, 'th': fr5_1, 'str': fr6_1, 'snr': fr7_1, 'prf': fr8_1,\
               'cnf': fr9_1, 'lc': fr10_1, 'snc': fr11_1, 'ctx': fr12_1, 'ei': GN1}  

    dbs_data = {"sn": fr1_2, "ge": fr2_2, 'gi': fr3_2, 'ppn': fr4_2, 'th': fr5_2, 'str': fr6_2, 'snr': fr7_2, 'prf': fr8_2,\
                'cnf': fr9_2, 'lc': fr10_2, 'snc': fr11_2, 'ctx': fr12_2, 'ei': GN2}  

    datalist = [h_data, pd_data, dbs_data]          
    
    ##write into a file
    pickle.dump(datalist, file)

    print(i+1)        
    
    
file.close()      

#%% Postprocessing
        
#Create an object generator function to load all the pickles
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#load pickles
items = loadall(name)   
c = list(items)
c_ar=np.array(c)

#initializing
sn_dbs=np.zeros(np.shape(c_ar)[1])
ge_dbs=np.zeros(np.shape(c_ar)[1])
gi_dbs=np.zeros(np.shape(c_ar)[1])
ppn_dbs=np.zeros(np.shape(c_ar)[1])
th_dbs=np.zeros(np.shape(c_ar)[1])
stri_dbs=np.zeros(np.shape(c_ar)[1])
snr_dbs=np.zeros(np.shape(c_ar)[1])
prf_dbs=np.zeros(np.shape(c_ar)[1])
cnf_dbs=np.zeros(np.shape(c_ar)[1])
lc_dbs=np.zeros(np.shape(c_ar)[1])
snc_dbs=np.zeros(np.shape(c_ar)[1])
ctx_dbs=np.zeros(np.shape(c_ar)[1])
ei_dbs=np.zeros(np.shape(c_ar)[1])

sn_pd=np.zeros(np.shape(c_ar)[1])
ge_pd=np.zeros(np.shape(c_ar)[1])
gi_pd=np.zeros(np.shape(c_ar)[1])
ppn_pd=np.zeros(np.shape(c_ar)[1])
th_pd=np.zeros(np.shape(c_ar)[1])
stri_pd=np.zeros(np.shape(c_ar)[1])
snr_pd=np.zeros(np.shape(c_ar)[1])
prf_pd=np.zeros(np.shape(c_ar)[1])
cnf_pd=np.zeros(np.shape(c_ar)[1])
lc_pd=np.zeros(np.shape(c_ar)[1])
snc_pd=np.zeros(np.shape(c_ar)[1])
ctx_pd=np.zeros(np.shape(c_ar)[1])
ei_pd=np.zeros(np.shape(c_ar)[1])

sn=np.zeros(np.shape(c_ar)[1])
ge=np.zeros(np.shape(c_ar)[1])
gi=np.zeros(np.shape(c_ar)[1])
ppn=np.zeros(np.shape(c_ar)[1])
th=np.zeros(np.shape(c_ar)[1])
stri=np.zeros(np.shape(c_ar)[1])
snr=np.zeros(np.shape(c_ar)[1])
prf=np.zeros(np.shape(c_ar)[1])
cnf=np.zeros(np.shape(c_ar)[1])
lc=np.zeros(np.shape(c_ar)[1])
snc=np.zeros(np.shape(c_ar)[1])
ctx=np.zeros(np.shape(c_ar)[1])
ei=np.zeros(np.shape(c_ar)[1])
#%%
#convenient formar
for i in range(np.shape(c_ar)[1]):
    sn_dbs[i]=c_ar[:,2][i]['sn']
    ge_dbs[i]=c_ar[:,2][i]['ge']
    gi_dbs[i]=c_ar[:,2][i]['gi']
    ppn_dbs[i]=c_ar[:,2][i]['ppn']
    th_dbs[i]=c_ar[:,2][i]['th']
    stri_dbs[i]=c_ar[:,2][i]['str']
    snr_dbs[i]=c_ar[:,2][i]['snr']
    prf_dbs[i]=c_ar[:,2][i]['prf']
    cnf_dbs[i]=c_ar[:,2][i]['cnf']
    lc_dbs[i]=c_ar[:,2][i]['lc']
    snc_dbs[i]=c_ar[:,2][i]['snc']
    ctx_dbs[i]=c_ar[:,2][i]['ctx']
    ei_dbs[i]=c_ar[:,2][i]['ei'] 

    sn_pd[i]=c_ar[:,1][i]['sn']
    ge_pd[i]=c_ar[:,1][i]['ge']
    gi_pd[i]=c_ar[:,1][i]['gi']
    ppn_pd[i]=c_ar[:,1][i]['ppn']
    th_pd[i]=c_ar[:,1][i]['th']
    stri_pd[i]=c_ar[:,1][i]['str']
    snr_pd[i]=c_ar[:,1][i]['snr']
    prf_pd[i]=c_ar[:,1][i]['prf']
    cnf_pd[i]=c_ar[:,1][i]['cnf']
    lc_pd[i]=c_ar[:,1][i]['lc']
    snc_pd[i]=c_ar[:,1][i]['snc']
    ctx_pd[i]=c_ar[:,1][i]['ctx']
    ei_pd[i]=c_ar[:,1][i]['ei']

    sn[i]=c_ar[:,0][i]['sn']
    ge[i]=c_ar[:,0][i]['ge']
    gi[i]=c_ar[:,0][i]['gi']
    ppn[i]=c_ar[:,0][i]['ppn']
    th[i]=c_ar[:,0][i]['th']
    stri[i]=c_ar[:,0][i]['str']
    snr[i]=c_ar[:,0][i]['snr']
    prf[i]=c_ar[:,0][i]['prf']
    cnf[i]=c_ar[:,0][i]['cnf']
    lc[i]=c_ar[:,0][i]['lc']
    snc[i]=c_ar[:,0][i]['snc']
    ctx[i]=c_ar[:,0][i]['ctx']
    ei[i]=c_ar[:,0][i]['ei']

#means and stds
ei_mean=np.mean(ei)
sn_mean=np.mean(sn)
ge_mean=np.mean(ge)
gi_mean=np.mean(gi)
ppn_mean=np.mean(ppn)
th_mean=np.mean(th)
stri_mean=np.mean(stri)
snr_mean=np.mean(snr)
prf_mean=np.mean(prf)
cnf_mean=np.mean(cnf)
lc_mean=np.mean(lc)
snc_mean=np.mean(snc)
ctx_mean=np.mean(ctx)

ei_std=np.std(ei)
sn_std=np.std(sn)
ge_std=np.std(ge)
gi_std=np.std(gi)
ppn_std=np.std(ppn)
th_std=np.std(th)
stri_std=np.std(stri)
snr_std=np.std(snr)
prf_std=np.std(prf)
cnf_std=np.std(cnf)
lc_std=np.std(lc)
snc_std=np.std(snc)
ctx_std=np.std(ctx)

ei_mean_pd=np.mean(ei_pd)
sn_mean_pd=np.mean(sn_pd)
ge_mean_pd=np.mean(ge_pd)
gi_mean_pd=np.mean(gi_pd)
ppn_mean_pd=np.mean(ppn_pd)
th_mean_pd=np.mean(th_pd)
stri_mean_pd=np.mean(stri_pd)
snr_mean_pd=np.mean(snr_pd)
prf_mean_pd=np.mean(prf_pd)
cnf_mean_pd=np.mean(cnf_pd)
lc_mean_pd=np.mean(lc_pd)
snc_mean_pd=np.mean(snc_pd)
ctx_mean_pd=np.mean(ctx_pd)

ei_std_pd=np.std(ei_pd)
sn_std_pd=np.std(sn_pd)
ge_std_pd=np.std(ge_pd)
gi_std_pd=np.std(gi_pd)
ppn_std_pd=np.std(ppn_pd)
th_std_pd=np.std(th_pd)
stri_std_pd=np.std(stri_pd)
snr_std_pd=np.std(snr_pd)
prf_std_pd=np.std(prf_pd)
cnf_std_pd=np.std(cnf_pd)
lc_std_pd=np.std(lc_pd)
snc_std_pd=np.std(snc_pd)
ctx_std_pd=np.std(ctx_pd)

ei_mean_dbs=np.mean(ei_dbs)
sn_mean_dbs=np.mean(sn_dbs)
ge_mean_dbs=np.mean(ge_dbs)
gi_mean_dbs=np.mean(gi_dbs)
ppn_mean_dbs=np.mean(ppn_dbs)
th_mean_dbs=np.mean(th_dbs)
stri_mean_dbs=np.mean(stri_dbs)
snr_mean_dbs=np.mean(snr_dbs)
prf_mean_dbs=np.mean(prf_dbs)
cnf_mean_dbs=np.mean(cnf_dbs)
lc_mean_dbs=np.mean(lc_dbs)
snc_mean_dbs=np.mean(snc_dbs)
ctx_mean_dbs=np.mean(ctx_dbs)

ei_std_dbs=np.std(ei_dbs)
sn_std_dbs=np.std(sn_dbs)
ge_std_dbs=np.std(ge_dbs)
gi_std_dbs=np.std(gi_dbs)
ppn_std_dbs=np.std(ppn_dbs)
th_std_dbs=np.std(th_dbs)
stri_std_dbs=np.std(stri_dbs)
snr_std_dbs=np.std(snr_dbs)
prf_std_dbs=np.std(prf_dbs)
cnf_std_dbs=np.std(cnf_dbs)
lc_std_dbs=np.std(lc_dbs)
snc_std_dbs=np.std(snc_dbs)
ctx_std_dbs=np.std(ctx_dbs)

#%% plotting
state = ['Healthy', 'PD', 'DBS']
x_pos = np.arange(len(state))

ei=np.array([ei_mean, ei_mean_pd, ei_mean_dbs])
ei_std_full=np.array([ei_std, ei_std_pd, ei_std_dbs])

lc=np.array([lc_mean, lc_mean_pd, lc_mean_dbs])
lc_std_full=np.array([lc_std, lc_std_pd, lc_std_dbs])

sn=np.array([sn_mean, sn_mean_pd, sn_mean_dbs])
sn_std_full=np.array([sn_std, sn_std_pd, sn_std_dbs])

ge=np.array([ge_mean, ge_mean_pd, ge_mean_dbs])
ge_std_full=np.array([ge_std, ge_std_pd, ge_std_dbs])

gi=np.array([gi_mean, gi_mean_pd, gi_mean_dbs])
gi_std_full=np.array([gi_std, gi_std_pd, gi_std_dbs])

ppn=np.array([ppn_mean, ppn_mean_pd, ppn_mean_dbs])
ppn_std_full=np.array([ppn_std, ppn_std_pd, ppn_std_dbs])

th=np.array([th_mean, th_mean_pd, th_mean_dbs])
th_std_full=np.array([th_std, th_std_pd, th_std_dbs])

stri=np.array([stri_mean, stri_mean_pd, stri_mean_dbs])
stri_std_full=np.array([stri_std, stri_std_pd, stri_std_dbs])

snr=np.array([snr_mean, snr_mean_pd, snr_mean_dbs])
snr_std_full=np.array([snr_std, snr_std_pd, snr_std_dbs])

prf=np.array([prf_mean, prf_mean_pd, prf_mean_dbs])
prf_std_full=np.array([prf_std, prf_std_pd, prf_std_dbs])

cnf=np.array([cnf_mean, cnf_mean_pd, cnf_mean_dbs])
cnf_std_full=np.array([cnf_std, cnf_std_pd, cnf_std_dbs])

snc=np.array([snc_mean, snc_mean_pd, snc_mean_dbs])
snc_std_full=np.array([snc_std, snc_std_pd, snc_std_dbs])

ctx=np.array([ctx_mean, ctx_mean_pd, ctx_mean_dbs])
ctx_std_full=np.array([ctx_std, ctx_std_pd, ctx_std_dbs])

def plot(mean, std, title, lable):
    fig, ax = plt.subplots()
    ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(lable)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state)
    ax.yaxis.grid(True)
    #plt.show()
    return fig
    
plot(ei, ei_std_full, "Error Index", "Error Index")
plot(sn, sn_std_full, "STN", "Firing rate, Hz")
plot(ge, ge_std_full, "GPe", "Firing rate, Hz")
plot(gi, gi_std_full, "GPi", "Firing rate, Hz")
plot(ppn, ppn_std_full, "PPN", "Firing rate, Hz")
plot(th, th_std_full, "Thalamus", "Firing rate, Hz")
plot(stri, stri_std_full, "Striatum", "Firing rate, Hz")
plot(lc, lc_std_full, "LC", "Firing rate, Hz")
plot(snr, snr_std_full, "SNr", "Firing rate, Hz")
plot(prf, prf_std_full, "PRF", "Firing rate, Hz")
plot(cnf, cnf_std_full, "CNF", "Firing rate, Hz")
plot(snc, snc_std_full, "SNc", "Firing rate, Hz")
plot(ctx, ctx_std_full, "Ctx", "Firing rate, Hz")

#%% plot all of them in one graph
#TBD

