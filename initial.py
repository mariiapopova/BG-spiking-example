# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:03:16 2021

@author: Maria
"""
import numpy as np
from createSMCinput import *
from FOGnetwork import *
from plotres import *
import time
%matplotlib auto


def Initial():
    
    start_time = time.time()
    #%% Set initial conditions
    
    #time variables
    tmax=1000              #maximum time (ms)
    dt=0.01                #timestep (ms)
    t=np.arange(0,tmax,dt) #time vector
    n=12                   #number of neurons in each nucleus (TH, STN, GPe, GPi)
    f=130
    
    #initial membrane voltages for all cells - random is a little different from matlab
    v1=-62+np.random.randn(1,n)*5
    v2=-62+np.random.randn(1,n)*5
    v3=-62+np.random.randn(1,n)*5
    v4=-62+np.random.randn(1,n)*5
    v5=-62+np.random.randn(1,n)*5 #for PPN
    v6=-62+np.random.randn(1,n)*5 #for SNR
    v7=-62+np.random.randn(1,n)*5 #for striatum #previous 63.8!
    v8=-62+np.random.randn(1,n)*5 #for PRF
    v9=-62+np.random.randn(1,n)*5 #for CNF
    v10=-62+np.random.randn(1,n)*5 #for LC
    v11=-62+np.random.randn(1,n)*5 #for SNc
    v12=-62+np.random.randn(1,n)*5 #for Ctx
    r=np.random.randn(1,n)*2 #what is r?
    
    
    #Sensorimotor cortex input to talamic cells
    Istim, timespike = createSMCinput(tmax,dt,14,0.2)
    #%% Running FOGnetwork
    #For 1000msec with 10 neurons in each nucleus, each condition will take roughly 60sec to run.
    
    #healthy
    vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx = FOGnetwork(0,0,0,Istim, timespike, tmax, dt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, r, n) #healthy
    h = plotres(vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx, t, timespike, tmax, Istim, dt)
    #pd
    vsn1, vge1, vgi1, vppn1, vth1, vstr1, vsnr1, vprf1, vcnf1, vlc1, vsnc1, vctx1 = FOGnetwork(1,0,0,Istim, timespike, tmax, dt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, r, n) #PD
    pd = plotres(vsn1, vge1, vgi1, vppn1, vth1, vstr1, vsnr1, vprf1, vcnf1, vlc1, vsnc1, vctx1, t,timespike,tmax,Istim,dt)
    #dbs
    vsn2, vge2, vgi2, vppn2, vth2, vstr2, vsnr2, vprf2, vcnf2, vlc2, vsnc2, vctx2 = FOGnetwork(1,1,f,Istim, timespike, tmax, dt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, r, n) #PD with DBS
    dbs = plotres(vsn2, vge2, vgi2, vppn2, vth2, vstr2, vsnr2, vprf2, vcnf2, vlc2, vsnc2, vctx2, t,timespike,tmax,Istim,dt)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return (vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx, \
           vsn1, vge1, vgi1, vppn1, vth1, vstr1, vsnr1, vprf1, vcnf1, vlc1, vsnc1, vctx1, \
           vsn2, vge2, vgi2, vppn2, vth2, vstr2, vsnr2, vprf2, vcnf2, vlc2, vsnc2, vctx2, \
           t, timespike, tmax)

if __name__ == "__main__":
    # execute only if run as a script
    fin_res=Initial()

