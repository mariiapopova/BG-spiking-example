# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:02:10 2021

@author: Maria
"""
import numpy as np
from gating import *
#from parameters import *
from createdbs import *
from dbssyn import *
from numba import jit
import matplotlib.pyplot as plt

#check maybe to also change gegi for pd sstate
@jit(nopython=True, cache=True)
def FOGnetwork(pd,stim,freq,Istim, timespike, tmax, dt, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, r, n):

#Usage: FOGnetwork(pd,stim,freq)
#
#Variables:
#pd - Variable to determine whether network is under the healthy or 
#Parkinsonian condition. For healthy, pd = 0, for Parkinson's, pd = 1.
#stim - Variable to determine whether deep brain stimulation is on.
#If DBS is off, stim = 0. If DBS is on, stim = 1.
#freq - Determines the frequency of stimulation, in Hz.
#  
#Author: Mariia Popova, UKE; based on Rosa So, Duke University 
#Updated 2/02/2021
     
    #loads initial conditions #what is r?
    ##Membrane parameters
    Cm=1 
    #In order of Th,STN,GP,PPN,Str,PRF,LC or Th,STN,GPe,GPi,PPN,SNr,Str,PRF,LC
    gl=np.array([0.05, 2.25, 0.1, 0.1, 0.4, 0.3]); El=np.array([-70, -60, -65, -67, -70, -17])
    gna=np.array([3, 37, 120, 30, 100, 120]); Ena=np.array([50, 55, 55, 45, 50, 45]) 
    gk=np.array([5, 45, 30, 3.2, 80, 10, 20]); Ek=np.array([-75, -80, -80, -95, -100, -95, -72])
    gt=np.array([5, 0.5, 0.5]); Et=0
    gca=np.array([0, 2, 0.15]); Eca=np.array([0, 140, 120])
    gahp=np.array([0, 20, 10]) #eahp and ek are the same excluding th
    gnal5=0.0207 #na leakage ppn
    gkl5=0.05    #k leakage ppn
    #gcort=0.15; Ecort=0      #cortex par for ppn
    gcort=0.15; Ecort=0      #cortex par for ppn
    ghyp5=0.4; Ehyp5=-43    #hyperpolarization-activated current ppn
    gnap5=1.9    #persistent sodium current ppn - chosen to have 8 Hz in rest
    Pca=1*10**(-3); z=2; F=96490; R=8.314; T=309.5; Cao=2; Cai=2.4e-4 #alike destexhe ghk for ppn
    Bcort=1 #ms^-1
    gm=1; Em=-100 #for striatum muscarinic current
    #lc params
    ga=47.7; Blc=0.21*(ga/gk[6])
    #stn coupling 
    gc1=0; gc2=0
    #snr coupling 
    gc3=0; gc4=0
    
    k1=np.array([0, 15, 10])    #dissociation const of ahp current 
    kca=np.array([0, 22.5, 15]) #calcium pump rate constant

    #synapse params alike in rubin SNr same to Gpi, PPN same to STN, Str same
    #to Gpi, CNF same to PPN, PRF same to PPN - why?, LC same to PPN - why?
    #SNc same to SNr - why?, Ctx same to LC????
    A=np.array([0, 3, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3]) 
    B=np.array([0, 0.1, 0.04, 0.08, 0.1, 0.08, 0.08, 0.1, 0.1, 0.1, 0.08, 0.1]) #maybe change 0.08 to so
    the=np.array([0, 30, 20, 20, 30, 20, 20, 30, 30, 30, 20, 30]) #maybe change alike so for prf cnf lc???
    
    ##Synapse parameters
    #In order of Igesn,Isnge,Igege,Isngi,Igegi,Igith 
    gsyn = np.array([1, 0.3, 1, 0.3, 1, .08])   #alike in Rubin gsyn and in So Esyn
    Esyn = np.array([-85, 0, -85, 0, -85, -85]) #alike in Rubin gsyn and in So Esyn

    tau=5; gpeak1=0.3; gpeak=0.43 #parameters for second-order alpha synapse

    gsynstr=np.array([0.8, 1.65, 17, 1.65, 0.05]); ggaba=0.1; gcorstr=0.07
    Esynstr=np.array([-85, 0, -85, -85, -85, -85, -85]); tau_i=13 #parameters for striatum synapses in order gaba-rec crtx strge gestr strgi strsnr snrstr

    # #gsynppn = np.array([0.061, 0.22, 0.061, 0.061, 0.22, 0.061, 0.061, 0.061, 0.061]) 
    # #gsynppn = np.array([0.061, 0.22, 0, 0, 0.22, 0, 0.061, 0.061, 0])
    # gsynppn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Esynppn = np.array([0, -85, 0, 0, -85, 0, 0, 0, 0]) #in order snppn gippn ppnsn ppngi snrppn ppnprf, prfppn, cnfppn, ppnge
    # #gsynppn = np.array([0, 0.17, 0.2, 0.18, 0.17, 0.18]); Esynppn = np.array([0, -85, 0, 0, -85, 0])
    # tau=5; gpeak1=0.3; gpeak=0.43 #parameters for second-order alpha synapse
    # #gsynsnr=np.array([0.15, 0.15, 0.15, 0.15, 0.15]) 
    # gsynsnr=np.array([0.061, 0.061, 0.061, 0.061, 0.061])
    # Esynsnr=np.array([0, 0, 0, 0, -85]) #for "to" snr synapses in order stn, prf, cnf, ppn, gpe
    # gsynstr=np.array([0.8, 1.65, 17, 1, 1]); ggaba=0.1; gcorstr=0.07
    # Esynstr=np.array([-85, 0, -85, -85, -85, -85, -85]); tau_i=13 #parameters for striatum synapses in order gaba-rec crtx strge gestr strgi strsnr snrstr
    # #gsyncnf=np.array([0.22, 0.22, 0.22]); Esyncnf=np.array([0,-85,0]) #cnf in order prf, snr, ppn
    # gsyncnf=np.array([0.061, 0.061, 0.061]); Esyncnf=np.array([0,-85,0]) #cnf in order prf, snr, ppn
    # gsynlc=np.array([0.061, 0.061, 0.061]); Esynlc=np.array([0, 0, 0]) #why these values?!!! - alike PPN? maybe change???
    # gsynprf = np.array([0.061, 0.061, 0.061]); Esynprf=np.array([0, -85, 0]) #why these values?!!!; prf in order cnf
    # gsynsnc = np.array([0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061]) 
    # Esynsnc=np.array([0, 0, -85, 0, -85, -85, -85, -85, -85, -85, -85, -85]) #snc in order lcsnc, ppnsnc, snrsnc, stnsnc, sncsnr, sncstn, sncgi, sncge, sncstr #why these values?!!!
    # gsynctx=np.array([0.061, 0.061, 0.061, 0.061]) #chosen how???? alike LC?
    # Esynctx=np.array([0,0,0,-85]) #ctx in order lcctx, prfctx, ppnctx, gectx

    Esyn_e = 0
    Esyn_i = -85

    gsyn_e = 0.15 #0.15
    gsyn_i = 0.8

    #time step
    t=np.arange(0,tmax,dt)

    ##Setting initial matrices  
    #n - number of neurons in each population
    vth=np.zeros(shape=(n,len(t)))   #thalamic membrane voltage
    vsn=np.zeros(shape=(n,len(t)))   #STN membrane voltage
    vge=np.zeros(shape=(n,len(t)))   #GPe membrane voltage
    vgi=np.zeros(shape=(n,len(t)))   #GPi membrane voltage
    vppn=np.zeros(shape=(n,len(t)))  #PPN membrane voltage
    vsnr=np.zeros(shape=(n,len(t)))  #SNr membrane voltage
    vstr=np.zeros(shape=(n,len(t)))  #striatum membrane voltage
    vprf=np.zeros(shape=(n,len(t)))  #prf membrane voltage
    vcnf=np.zeros(shape=(n,len(t)))  #cnf membrane voltage
    vlc=np.zeros(shape=(n,len(t)))   #LC membrane voltage
    vsnc=np.zeros(shape=(n,len(t)))  #SNc membrane voltage
    vctx=np.zeros(shape=(n,len(t)))  #Ctx membrane voltage
    vef=np.zeros(shape=(n,len(t)))   #STN effs membrane voltage
    vsnre=np.zeros(shape=(n,len(t))) #SNr effs membrane voltage

    Z4=np.zeros(n)  #for 2 order alpha-synapse gpi-th current
    S4=np.zeros(n)  #for alpha-synapse gpi-th current
    S3=np.zeros(n)  #for alpha-synapse ge-ge/sn/str/snr/snc current
    S3_1=np.zeros(n) #for dummy gesn current
    S2=np.zeros(n)  #for alpha-synapse snge/gi/snr current
    Z2=np.zeros(n)  #for 2 order alpha-synapse sn current
    S2_1=np.zeros(n) #for dummy snge current
    S3_2=np.zeros(n) #for dummy gege current
    S5=np.zeros(n)  #for alpha-synapse stn-ppn/snr/prf/snc current
    S6=np.zeros(n)  #for alpha-synapse gpi-ppn/snc current
    S7=np.zeros(n)  #for alpha-synapse ppn-stn/gi/prf/cnf/snr/ge/snc current
    S8=np.zeros(n)  #for alpha-synapse dummy out of ctx current
    Sc=np.zeros(shape=(n,len(t))) #for cortex-ppn synapse
    S1c=np.zeros(n) #for striatum gaba-rec
    #for synapses striatum-gaba-rec
    gall=np.random.choice(np.arange(n),n,replace=False)
    gbll=np.random.choice(np.arange(n),n,replace=False)
    gcll=np.random.choice(np.arange(n),n,replace=False)
    gdll=np.random.choice(np.arange(n),n,replace=False)
    S10=np.zeros(n) #for alpha-synapse str-ge/gi/snr/snc current
    S12=np.zeros(n) #for alpha-synapse snc-snr/stn/gpi/gpe/str current
    S14=np.zeros(n) #for alpha-synapse snr-ppn/cnf/prf/str/snc current
    S18=np.zeros(n) #for alpha-synapse cnf-lc/prf/ppn/snr current
    S19=np.zeros(n) #for alpha-synapse prf-lc/cnf/ppn/snr current
    S20=np.zeros(n) #for alpha-synapse lc-prf/snc current

    ##with or without DBS
    if stim==0: 
        Idbs=np.zeros(len(t))
        Idbs1=np.zeros(len(t))
    else:
        #Idbs=createdbs(freq,tmax,dt) #creating DBS train with frequency freq
        Idbs=dbssyn(freq,tmax,dt) #creating DBS train with frequency 
        Idbs1=dbssyn(60,tmax,dt) #creating DBS train with frequency freq, low-freq for ssnr

    ##initial conditions 
    vth[:,0]=v1
    vsn[:,0]=v2
    vge[:,0]=v3
    vgi[:,0]=v4
    vppn[:,0]=v5  #for PPN
    vsnr[:,0]=v6  #for SNr     
    vstr[:,0]=v7  #for striatum D2
    vprf[:,0]=v8  #for PRF
    vcnf[:,0]=v9  #for CNF  
    vlc[:,0]=v10  #for LC
    vsnc[:,0]=v11  #for SNc
    vctx[:,0]=v12  #for ctx
    vef[:,0]=v2   #for STN ef
    vsnre[:,0]=v6 #for SNr ef

    #helper variables for gating and synapse params - starting parameters
    R2=stn_rinf(vsn[:,0])        #r for stn
    H1=th_hinf(vth[:,0])         #h for th
    R1=th_rinf(vth[:,0])         #r for th
    H12=th_hinf(vctx[:,0])         #h for ctx
    R12=th_rinf(vctx[:,0])         #r for ctx
    N2=stn_ninf(vsn[:,0])        #n for stn
    H2=stn_hinf(vsn[:,0])        #h for stn
    C2=stn_cinf(vsn[:,0])        #c for stn
    CA2=np.array([0.1])          #intracellular concentration of Ca2+ in muM for stn
    CA3=CA2                      #for gpe
    CA4=CA2                      #for gpi
    CA6=CA2                      #for snr
    CA11=CA2                     #for snc
    CA9=CA2                      #for cnf
    N3=gpe_ninf(vge[:,0])        #n for gpe
    H3=gpe_hinf(vge[:,0])        #h for gpe
    R3=gpe_rinf(vge[:,0])        #r for gpe
    N4=gpe_ninf(vgi[:,0])        #n for gpi
    H4=gpe_hinf(vgi[:,0])        #h for gpi
    R4=gpe_rinf(vgi[:,0])        #r for gpi
    M5=ppn_minf(vppn[:,0])       #m for ppn na
    H5=ppn_hinf(vppn[:,0])       #h for ppn na
    Mk5=ppn_mkinf(vppn[:,0])     #m for ppn k
    Mh5=ppn_mhinf(vppn[:,0])     #m for ppn hyp
    Hnap5=ppn_hnapinf(vppn[:,0]) #h for ppn nap
    Mt5=ppn_mtinf(vppn[:,0])     #m for ppn low-tresh
    Ht5=ppn_htinf(vppn[:,0])     #h for ppn low-tresh
    N6=gpe_ninf(vsnr[:,0])       #n for snr
    H6=gpe_hinf(vsnr[:,0])       #h for snr
    R6=gpe_rinf(vsnr[:,0])       #r for snr
    N8=gpe_ninf(vprf[:,0])       #n for prf
    H8=gpe_hinf(vprf[:,0])       #h for prf
    R8=gpe_rinf(vprf[:,0])       #r for prf
    N9=gpe_ninf(vcnf[:,0])      #n for snc
    H9=gpe_hinf(vcnf[:,0])      #h for snc
    R9=gpe_rinf(vcnf[:,0])      #r for snc
    # M9=ppn_minf(vcnf[:,0])       #m for cnf na
    # H9=ppn_hinf(vcnf[:,0])       #h for cnf na
    # Mk9=ppn_mkinf(vcnf[:,0])     #m for cnf k
    # Mh9=ppn_mhinf(vcnf[:,0])     #m for cnf hyp
    # Hnap9=ppn_hnapinf(vcnf[:,0]) #h for cnf nap
    # Mt9=ppn_mtinf(vcnf[:,0])     #m for cnf low-tresh
    # Ht9=ppn_htinf(vcnf[:,0])     #h for cnf low-tresh
    q=lc_qinf(vlc[:,0])          #q for lc
    N11=gpe_ninf(vsnc[:,0])      #n for snc
    H11=gpe_hinf(vsnc[:,0])      #h for snc
    R11=gpe_rinf(vsnc[:,0])      #r for snc
    #for STN efferents
    N2e=stn_ninf(vef[:,0])
    H2e=stn_hinf(vef[:,0])
    R2e=stn_rinf(vef[:,0])
    CA2e=CA2
    C2e=stn_cinf(vef[:,0])
    #for SNr efferents
    N6e=gpe_ninf(vsnre[:,0])     #n for snr
    H6e=gpe_hinf(vsnre[:,0])     #h for snr
    R6e=gpe_rinf(vsnre[:,0])     #r for snr
    CA6e=CA2                     #for snr

    #striatum gating
    m7=str_alpham(vstr[:,0])/(str_alpham(vstr[:,0])+str_betam(vstr[:,0]))
    h7=str_alphah(vstr[:,0])/(str_alphah(vstr[:,0])+str_betah(vstr[:,0]))
    n7=str_alphan(vstr[:,0])/(str_alphan(vstr[:,0])+str_betan(vstr[:,0]))
    p7=str_alphap(vstr[:,0])/(str_alphap(vstr[:,0])+str_betap(vstr[:,0]))

    timespikeint=np.ones_like(timespike)
    timespikeint = (np.round(timespike,0,timespikeint)).astype(np.int64) #index when 1 for ppn
    looptimespikeint=(timespikeint/dt).astype(np.int64)
    
    ##Time loop
    for i in range(1, len(t)):
    
        #condition for cortex current for ppn and striatum
        if np.sum(looptimespikeint==i)==1:
            Sc[:,i-1] = 1
    
        #previous values
        V1=vth[:,i-1];    V2=vsn[:,i-1];     V3=vge[:,i-1];    V4=vgi[:,i-1];   V5=vppn[:,i-1] 
        V6=vsnr[:,i-1];   V7=vstr[:,i-1];    V8=vprf[:,i-1];   V9=vcnf[:,i-1];  V10=vlc[:,i-1]
        V11=vsnc[:,i-1];  V12=vctx[:,i-1]
        Ve=vef[:,i-1];    Vsnre=vsnre[:,i-1]
        
        #Synapse parameters 
        S2_1[1:n]=S2[0:n-1];S2_1[0]=S2[n-1]    #dummy synapse for snge current as there is 1 stn to 2 ge
        S3_1[0:n-1]=S3[1:n];S3_1[-1]=S3[0]     #dummy synapse for gesn current as there is 1 ge to 2 stn
        S3_2[2:n]=S3[0:n-2];S3_2[:2]=S3[n-2:n] #dummy synapse for gege current as there is 1 ge to 2 ge
        S11cr=S1c[gall];S12cr=S1c[gbll];S13cr=S1c[gcll];S14cr=S1c[gdll] #dummy striatum crtx current
        
        #membrane parameters - gating variables
        m1=th_minf(V1);  m2=stn_minf(V2); m3=gpe_minf(V3); m4=gpe_minf(V4)
        m6=gpe_minf(V6); m8=gpe_minf(V8); m9=gpe_minf(V9); m11=gpe_minf(V11)
        m12=th_minf(V12) #gpe and gpi are modeled similarily

        n2=stn_ninf(V2); n3=gpe_ninf(V3); n4=gpe_ninf(V4); n6=gpe_ninf(V6); n8=gpe_ninf(V8) 
        n9=gpe_ninf(V9); n11=gpe_ninf(V11)

        h1=th_hinf(V1);  h2=stn_hinf(V2); h3=gpe_hinf(V3); h4=gpe_hinf(V4) 
        h6=gpe_hinf(V6); h9=gpe_hinf(V9); h11=gpe_hinf(V11); h12=th_hinf(V12)
        h8=gpe_hinf(V8)

        p1=th_pinf(V1); p12=th_pinf(V12)

        a2=stn_ainf(V2); a3=gpe_ainf(V3); a4=gpe_ainf(V4); a6=gpe_ainf(V6)
        a8=gpe_ainf(V8); a9=gpe_ainf(V9); a11=gpe_ainf(V11) #for low-treshold ca

        b2=stn_binf(R2)

        s3=gpe_sinf(V3); s4=gpe_sinf(V4); s6=gpe_sinf(V6); s9=gpe_sinf(V9)
        s11=gpe_sinf(V11)

        r1=th_rinf(V1);  r2=stn_rinf(V2); r3=gpe_rinf(V3); r4=gpe_rinf(V4); r6=gpe_rinf(V6)
        r9=gpe_rinf(V9); r11=gpe_rinf(V11); r12=th_rinf(V12); r8=gpe_rinf(V8)

        c2=stn_cinf(V2)

        #for ppn
        m5=ppn_minf(V5); h5=ppn_hinf(V5); mk5=ppn_mkinf(V5); mh5=ppn_mhinf(V5); 
        mnap5=ppn_mnapinf(V5); hnap5=ppn_hnapinf(V5); mt5=ppn_mtinf(V5); ht5=ppn_htinf(V5)
        # #for cnf
        # m9=ppn_minf(V9); h9=ppn_hinf(V9); mk9=ppn_mkinf(V9); mh9=ppn_mhinf(V9); mnap9=ppn_mnapinf(V9)
        # hnap9=ppn_hnapinf(V9); mt9=ppn_mtinf(V9); ht9=ppn_htinf(V9)
        #for lc
        qinf=lc_qinf(V10); m10=lc_minf(V10); b10=lc_binf(V10)
        #for effs
        m2e=stn_minf(Ve); n2e=stn_ninf(Ve); h2e=stn_hinf(Ve); a2e=stn_ainf(Ve); b2e=stn_binf(R2e)
        r2e=stn_rinf(Ve); c2e=stn_cinf(Ve)
        m6e=gpe_minf(Vsnre); n6e=gpe_ninf(Vsnre); h6e=gpe_hinf(Vsnre); a6e=gpe_ainf(Vsnre)
        s6e=gpe_sinf(Vsnre); r6e=gpe_rinf(Vsnre)
    
        #membrane parameters - time constants
        tn2=stn_taun(V2); tn3=gpe_taun(V3); tn4=gpe_taun(V4); tn6=gpe_taun(V6) 
        tn8=gpe_taun(V8); tn9=gpe_taun(V9); tn11=gpe_taun(V11)

        th1=th_tauh(V1); th2=stn_tauh(V2); th3=gpe_tauh(V3); th4=gpe_tauh(V4)
        th6=gpe_tauh(V6); th9=gpe_tauh(V9); th11=gpe_tauh(V11); th8=gpe_tauh(V8)
        th12=th_tauh(V12)

        tr1=th_taur(V1); tr2=stn_taur(V2); tr3=30; tr4=30; tr6=30; tr8=30
        tr9=30; tr11=30; tr12=th_taur(V12)

        tc2=stn_tauc(V2); 

        #for ppn
        tm5=ppn_taum(V5); th5=ppn_tauh(V5); tmk5=ppn_taumk(V5); tmh5=ppn_taumh(V5)
        thnap5=ppn_tauhnap(V5); tmt5=ppn_taumt(V5); tht5=ppn_tauht(V5)
        # #for cnf
        # tm9=ppn_taum(V9); th9=ppn_tauh(V9); tmk9=ppn_taumk(V9); tmh9=ppn_taumh(V9)
        # thnap9=ppn_tauhnap(V9); tmt9=ppn_taumt(V9); tht9=ppn_tauht(V9)
        #for lc
        tq=lc_tauq(V10)
        #for effs
        tn2e=stn_taun(Ve); th2e=stn_tauh(Ve); tr2e=stn_taur(Ve); tc2e=stn_tauc(Ve)
        tn6e=gpe_taun(Vsnre); th6e=gpe_tauh(Vsnre); tr6e=30
    
        #thalamic cell currents
        Il1=gl[0]*(V1-El[0])
        Ina1=gna[0]*(m1**3)*H1*(V1-Ena[0])
        Ik1=gk[0]*((0.75*(1-H1))**4)*(V1-Ek[0]) #misspelled in So paper
        It1=gt[0]*(p1**2)*R1*(V1-Et)
        Igith=1.4*gsyn[5]*(V1-Esyn[5])*S4 #for alpha-synapse second order kinetics
    
        #STN cell currents
        Il2=gl[1]*(V2-El[1])
        Ik2=gk[1]*(N2**4)*(V2-Ek[1])
        Ina2=gna[1]*(m2**3)*H2*(V2-Ena[1])
        It2=gt[1]*(a2**3)*(b2**2)*(V2-Eca[1]) #misspelled in So paper
        Ica2=gca[1]*(C2**2)*(V2-Eca[1])
        Iahp2=gahp[1]*(V2-Ek[1])*(CA2/(CA2+k1[1])) #cause ek and eahp are the same
        Igesn=0.5*(gsyn[0]*(V2-Esyn[0])*(S3+S3_1))  #first-order kinetics 1ge to 2sn
        #Iappstn=36
        Iappstn=35 #38.5
        Ippnsn=0.1*(V2-Esyn_e)*S7 #first-order kinetics ppn to stn
        Isncsn=(1-pd)*gsyn_i*(V2-Esyn_i)*S12 #first-order kinetics snc to stn
        Icorsn=0.5*gcort*(Sc[:,i-1]+S8)*(V2-Ecort) #alike ppn
        Ic1=gc1*(V2-Ve)
        #efferents
        Il2e=gl[1]*(Ve-El[1])
        Ik2e=gk[1]*(N2e**4)*(Ve-Ek[1])
        Ina2e=gna[1]*(m2e**3)*H2e*(Ve-Ena[1])
        It2e=gt[1]*(a2e**3)*(b2e**2)*(Ve-Eca[1])
        Ica2e=gca[1]*(C2e**2)*(Ve-Eca[1])
        Iahp2e=gahp[1]*(Ve-Ek[1])*(CA2e/(CA2e+k1[1]))
        Ic2=gc2*(Ve-V2)
        Igesne=0.5*(gsyn[0]*(Ve-Esyn[0])*(S3+S3_1)) #first-order kinetics 1ge to 2sn
        Icorsne=0.5*gcort*(Sc[:,i-1]+S8)*(Ve-Ecort) #alike ppn
        Ippnsne=0.1*(Ve-Esyn_e)*S7 #first-order kinetics ppn to stn
        Isncsne=(1-pd)*gsyn_i*(Ve-Esyn_i)*S12 #first-order kinetics snc to stn
    
        #GPe cell currents
        Il3=gl[2]*(V3-El[2])
        Ik3=gk[2]*(N3**4)*(V3-Ek[2])  
        Ina3=gna[2]*(m3**3)*H3*(V3-Ena[2])
        It3=gt[2]*(a3**3)*R3*(V3-Eca[2]) #Eca as in Rubin and Terman
        Ica3=gca[2]*(s3**2)*(V3-Eca[2])  #misspelled in So paper
        Iahp3=gahp[2]*(V3-Ek[2])*(CA3/(CA3+k1[2])) #as Ek is the same with Eahp
        Isnge=0.5*(gsyn[1]*(V3-Esyn[1])*(S2+S2_1)) #second-order kinetics 1sn to 2ge
        #Igege=0.5*((gsyn[2]+8*pd)*(V3-Esyn[2])*(S3_1+S3_2))
        Igege=0.5*((gsyn[2]+3*pd)*(V3-Esyn[2])*(S3_1+S3_2))
        Iappgpe=16.5 #15#17
        #str-gpe synapse 
        Istrge=gsynstr[0]*(V3-Esynstr[2])*S10 #1str to 1ge
        #ppn-gpe synapse 
        Ippnge=gsyn_e*(V3-Esyn_e)*S7 #1ppn to 1ge
        #snc-gpe synapse 
        Isncge=(1-pd)*gsyn_i*(V3-Esyn_i)*S12 #1snc to 1ge
        #cortical connection
        Icorge=0.5*gcort*(Sc[:,i-1]+S8)*(V3-Ecort) #alike ppn
        
        #GPi cell currents
        Il4=gl[2]*(V4-El[2])
        Ik4=gk[2]*(N4**4)*(V4-Ek[2])
        Ina4=gna[2]*(m4**3)*H4*(V4-Ena[2]) #Eca as in Rubin and Terman
        It4=gt[2]*(a4**3)*R4*(V4-Eca[2])   #misspelled in So paper
        Ica4=gca[2]*(s4**2)*(V4-Eca[2]) 
        Iahp4=gahp[2]*(V4-Ek[2])*(CA4/(CA4+k1[2])) #as Ek is the same with Eahp
        Isngi=0.5*(gsyn[3]*(V4-Esyn[3])*(S2+S2_1)) #second-order kinetics 1sn to 2gi
        Igegi=0.5*(gsyn[4]*(V4-Esyn[4])*(S3_1+S3_2)) #first-order kinetics 1ge to 2gi
        Iappgpi=17 #17
        Ippngi=gsyn_e*(V4-Esyn_e)*S7 #first-order kinetics ppn to gpi
        #str-gpi synapse
        Istrgi=gsynstr[2]*(V4-Esynstr[4])*S10 #1str to 1gi
        #snc-gpi synapse
        Isncgi=(1-pd)*gsyn_i*(V4-Esyn_i)*S12 #1snc to 1gi
    
        #PPN cell currents  
        Inal5=gnal5*(V5-Ena[3])
        Ikl5=gkl5*(V5-Ek[3])
        Ina5=gna[3]*(M5**3)*H5*(V5-Ena[3])
        Ik5=gk[3]*(Mk5**4)*(V5-Ek[3])
        Ihyp5=ghyp5*(Mh5**3)*(V5-Ehyp5)
        #alike Rubin           
        Inap5=gnap5*mnap5*Hnap5*(V5-Ena[3]) 
        #It alike Destexhe   
        zet=Pca*F*z*V5/(R*T)     
        if (np.abs(zet)>1e-4).all():
            Gt5=(Pca*z*F)*(Cai*(-zet/(np.exp(-zet)-1))-Cao*(zet/(np.exp(zet)-1)))
        else:
            Gt5=(Pca*z*F)*(Cai*(1+zet/2)-Cao*(1-zet/2))
        It5=.2e-3*Mt5**2*Ht5*Gt5 #alike destexhe
        Iappppn=0 #chosen from fig.5 in Lourens
        Isnppn=0.1*(V5-Esyn_e)*S5 #first-order kinetics stn to ppn
        Igippn=0.65*(V5-Esyn_i)*S6 #first-order kinetics gpi to ppn
        Icort=0.5*gcort*(Sc[:,i-1]+S8)*(V5-Ecort)
        #Icort=0
        Isnrppn=0.65*(V5-Esyn_i)*S14 #first-order kinetics snr to ppn
        Iprfppn=gsyn_e*(V5-Esyn_e)*S19 #first-order kinetics prf to ppn
        Icnfppn=gsyn_e*(V5-Esyn_e)*S18 #first-order kinetics cnf to ppn
    
        #SNr cell currents - modelled as GPi
        Il6=gl[2]*(V6-El[2])
        Ik6=gk[2]*(N6**4)*(V6-Ek[2])
        Ina6=gna[2]*(m6**3)*H6*(V6-Ena[2]) #Eca as in Rubin and Terman
        It6=gt[2]*(a6**3)*R6*(V6-Eca[2])   #misspelled in So paper
        Ica6=gca[2]*(s6**2)*(V6-Eca[2]) 
        Iahp6=gahp[2]*(V6-Ek[2])*(CA6/(CA6+k1[2])) #as Ek is the same with Eahp
        #Isnsnr=gsynsnr[0]*(V6-Esynsnr[0])*S5 #first-order kinetics 1sn to 1snr
        Isnsnr=0.5*(gsyn_e*(V6-Esyn_e)*(S2+S2_1))
        #str-snr synapse
        Istrsnr=gsynstr[3]*(V6-Esynstr[5])*S10 #1str to 1snr
        #other synapses
        Iprfsnr=0.45*(V6-Esyn_e)*S19 #1prf to 1snr #0.6
        Icnfsnr=0.45*(V6-Esyn_e)*S18 #1cnf to 1snr
        Ippnsnr=gsyn_e*(V6-Esyn_e)*S7 #1ppn to 1snr
        Igesnr=gsyn_i*(V6-Esyn_i)*S3 #1ge to 1snr
        Isncsnr=(1-pd)*gsyn_i*(V6-Esyn_i)*S12 #1snc to 1snr
        Iappsnr=0 #0
        Ic3=gc3*(V6-Vsnre)
        #efferents
        Il6e=gl[2]*(Vsnre-El[2])
        Ik6e=gk[2]*(N6e**4)*(Vsnre-Ek[2])
        Ina6e=gna[2]*(m6e**3)*H6e*(Vsnre-Ena[2]) #Eca as in Rubin and Terman
        It6e=gt[2]*(a6e**3)*R6e*(Vsnre-Eca[2])   #misspelled in So paper
        Ica6e=gca[2]*(s6e**2)*(Vsnre-Eca[2]) 
        Iahp6e=gahp[2]*(Vsnre-Ek[2])*(CA6e/(CA6e+k1[2])) #as Ek is the same with Eahp
        #Isnsnre=gsynsnr[0]*(V6-Esynsnr[0])*S5 #first-order kinetics 1sn to 1snr
        Isnsnre=0.5*(gsyn_e*(Vsnre-Esyn_e)*(S2+S2_1))
        #str-snr synapse
        Istrsnre=gsynstr[3]*(Vsnre-Esynstr[5])*S10 #1str to 1ge
        #other synapses
        Iprfsnre=0.45*(Vsnre-Esyn_e)*S19 #1prf to 1snr #0.6
        Icnfsnre=0.45*(Vsnre-Esyn_e)*S18 #1cnf to 1snr
        Ippnsnre=gsyn_e*(Vsnre-Esyn_e)*S7 #1ppn to 1snr
        Igesnre=gsyn_i*(Vsnre-Esyn_i)*S3 #1ge to 1snr
        Isncsnre=(1-pd)*gsyn_i*(Vsnre-Esyn_i)*S12 #1snc to 1snr
        Ic4=gc4*(Vsnre-V6)
  
        #Striatum D2 cell currents
        Ina7=gna[4]*(m7**3)*h7*(V7-Ena[4])
        Ik7=gk[4]*(n7**4)*(V7-Ek[4])
        Il7=gl[3]*(V7-El[3])
        #Im7=(2.6-2.5*pd)*gm*p7*(V7-Em) 
        Im7=(2.6-0.2*pd)*gm*p7*(V7-Em) 
        #Im7=(2.6-bopt*pd)*gm*p7*(V7-Em)
        Igaba7=(ggaba/4)*(V7-Esynstr[0])*(S11cr+S12cr+S13cr+S14cr) #maybe change to 3.5 for direct and indirect #recieves input from 40% remaining
        #Icorstr=(6*gcorstr-0.3*pd)*(V7-Esynstr[1])*Sc[:,i-1] #optimized
        Icorstr=0.5*(5*gcorstr-0.3*pd)*(V7-Esynstr[1])*(Sc[:,i-1]+S8) #optimized 
        #ge-str synapse
        Igestr=gsynstr[1]*(V7-Esynstr[3])*S3 #1ge to 1str
        #snr-str synapse
        Isnrstr=gsynstr[4]*(V7-Esynstr[6])*S14 #1snr to 1str
        #snc-str synapse
        Isncstr=(1-pd)*gsyn_i*(V7-Esyn_i)*S12 #1snc to 1str, make as snr
        Iappstr=3.5
    
        #PRF cell currents - alike gpe
        Il8=gl[4]*(V8-El[4])  
        Ik8=gk[5]*(N8**4)*(V8-Ek[5])
        Ina8=gna[5]*(m8**3)*H8*(V8-Ena[5]) #as in Miura constants
        #Il8=gl[2]*(V8-El[2])
        #Ik8=gk[2]*(N8**4)*(V8-Ek[2])
        #Ina8=gna[2]*(m8**3)*H8*(V8-Ena[2]) #as in So constants
        It8=gt[2]*(a8**3)*R8*(V8-Eca[2]) #Eca as in Rubin and Terman 
        Icorprf=0.5*gcort*(Sc[:,i-1]+S8)*(V8-Ecort) #alike ppn #0.05
        Iappprf=0
        Ippnprf=gsyn_e*(V8-Esyn_e)*S7 #first-order kinetics ppn to gpi alike its not cholinergic
        Ilcprf=0.15*(V8-Esyn_e)*S20
        Icnfprf=0.15*(V8-Esyn_e)*S18 #0.05
        Isnrprf=gsyn_i*(V8-Esyn_i)*S14 #3!!
        Istnprf=gsyn_e*(V8-Esyn_e)*S5
    
        #CNF cell currents - alike GPi
        # Inal9=gnal5*(V9-Ena[3])
        # Ikl9=gkl5*(V9-Ek[3])
        # Ina9=gna[3]*(M9**3)*H9*(V9-Ena[3])
        # Ik9=gk[3]*(Mk9**4)*(V9-Ek[3])
        # Ihyp9=ghyp5*(Mh9**3)*(V9-Ehyp5)
        # #alike Rubin
        # Inap9=gnap5*mnap9*Hnap9*(V9-Ena[3]) 
        # #It alike Destexhe
        # zet1=Pca*F*z*V9/(R*T)
        # if (np.abs(zet1)>1e-4).all():
        #     Gt9=(Pca*z*F)*(Cai*(-zet1/(np.exp(-zet1)-1))-Cao*(zet1/(np.exp(zet1)-1)))
        # else:
        #     Gt9=(Pca*z*F)*(Cai*(1+zet1/2)-Cao*(1-zet1/2))
        # It9=.2e-3*Mt9**2*Ht9*Gt9 #alike destexhe
        Il9=gl[2]*(V9-El[2])
        Ik9=gk[2]*(N9**4)*(V9-Ek[2])
        Ina9=gna[2]*(m9**3)*H9*(V9-Ena[2]) #Eca as in Rubin and Terman
        It9=gt[2]*(a9**3)*R9*(V9-Eca[2])   #misspelled in So paper
        Ica9=gca[2]*(s9**2)*(V9-Eca[2]) 
        Iahp9=gahp[2]*(V9-Ek[2])*(CA9/(CA9+k1[2])) #as Ek is the same with Eahp
        Iappcnf=0
        Iprfcnf=gsyn_e*(V9-Esyn_e)*S19 #first-order kinetics prf to cnf #0.05
        Isnrcnf=2*(V9-Esyn_i)*S14 #first-order kinetics snr to cnf #3!!
        Ippncnf=gsyn_e*(V9-Esyn_e)*S7 #first-order kinetics ppn to cnf
    
        #LC cell currents
        Ina10=gna[2]*(m10**3)*(-3*(q-Blc*b10)+0.85)*(V10-Ena[2])
        Ik10=gk[6]*q*(V10-Ek[6])  
        Il10=gl[5]*(V10-El[5])
        Iapplc=0.38#4.2#5    #0.4??
        Icnflc=gsyn_e*(V10-Esyn_e)*S18 
        Iprflc=gsyn_e*(V10-Esyn_e)*S19  

        #SNc cell currents - modelled as GPi
        Il11=gl[2]*(V11-El[2])
        Ik11=gk[2]*(N11**4)*(V11-Ek[2])
        Ina11=gna[2]*(m11**3)*H11*(V11-Ena[2]) #Eca as in Rubin and Terman
        It11=gt[2]*(a11**3)*R11*(V11-Eca[2])   #misspelled in So paper
        Ica11=gca[2]*(s11**2)*(V11-Eca[2]) 
        Iahp11=gahp[2]*(V11-Ek[2])*(CA11/(CA11+k1[2])) #as Ek is the same with Eahp
        #synapses
        Icorsnc=0.5*gcort*(Sc[:,i-1]+S8)*(V11-Ecort) #alike ppn
        Ilcsnc=gsyn_e*(V11-Esyn_e)*S20 #1lc to 1snc
        Ippnsnc=gsyn_e*(V11-Esyn_e)*S7 #1ppn to 1snc
        Isnrsnc=gsyn_i*(V11-Esyn_i)*S14 #1snr to 1snc
        Isnsnc=gsyn_e*(V11-Esyn_e)*S5 #1stn to 1snc
        Igisnc=gsyn_i*(V11-Esyn_i)*S6 #1gi to 1snc
        Igesnc=gsyn_i*(V11-Esyn_i)*S3 #1ge to 1snc
        Istrsnc=gsyn_i*(V11-Esyn_i)*S10 #1str to 1snc
        Iappsnc=3 

        #Ctx cell currents - no HH dynamics for now
        Ilcctx=gsyn_e*(V12-Esyn_e)*S20 #1lc to 1ctx
        Iprfctx=gsyn_e*(V12-Esyn_e)*S19 #1prf to 1ctx
        Ippnctx=gsyn_e*(V12-Esyn_e)*S7 #1ppn to 1 ctx
        Igectx=gsyn_i*(V12-Esyn_i)*S3 #1gpe to 1ctx
        Il12=gl[0]*(V12-El[0])
        Ina12=gna[0]*(m12**3)*H12*(V12-Ena[0])
        Ik12=gk[0]*((0.75*(1-H12))**4)*(V12-Ek[0]) #misspelled in So paper
        It12=gt[0]*(p12**2)*R12*(V12-Et)
        Iappctx=0

    
        #Differential Equations for cells using forward Euler method
        
        #thalamic
        vth[:,i]=V1+dt*(1/Cm*(-Il1-Ik1-Ina1-It1-Igith+Istim[i]))
        H1=H1+dt*((h1-H1)/th1)
        R1=R1+dt*((r1-R1)/tr1)

        #for cortex
        vctx[:,i]=V12+dt*(1/Cm*(-Il12-Ik12-Ina12-It12-Ilcctx-Iprfctx-Ippnctx-Igectx+Iappctx)) #do we need istim?
        H12=H12+dt*((h12-H12)/th12)
        R12=R12+dt*((r12-R12)/tr12)
        Sc[:,i]=Sc[:,i-1]+dt*(-Bcort*Sc[:,i-1])
        S8=S8+dt*(A[11]*(1-S8)*Hinf(V12-the[11])-B[11]*S8)
        
    
        #STN
        #vsn[:,i]=V2+dt*(1/Cm*(-Il2-Ik2-Ina2-It2-Ica2-Iahp2-Igesn-Isncsn+Iappstn+Idbs[i]-Ippnsn)) #currently STN-DBS
        ch = V2+dt*(1/Cm*(-Il2-Ik2-Ina2-It2-Ica2-Iahp2-Igesn-Isncsn+Iappstn-Ippnsn-Icorsn-Ic1))
        if len(ch[np.logical_and(ch>-10, Idbs[i]>10)])!=0:
            ch[np.logical_and(ch>-10, Idbs[i]>10)]=ch[np.logical_and(ch>-10, Idbs[i]>10)]
        ch[np.logical_not(np.logical_and(ch>-10, Idbs[i]>10))]=\
            ch[np.logical_not(np.logical_and(ch>-10, Idbs[i]>10))]+dt*(1/Cm*(0.1*Idbs[i])) #why 0.01???
        vsn[:,i]=ch
        N2=N2+dt*(0.75*(n2-N2)/tn2)
        H2=H2+dt*(0.75*(h2-H2)/th2)
        R2=R2+dt*(0.2*(r2-R2)/tr2)
        CA2=CA2+dt*(3.75*1e-5*(-Ica2-It2-kca[1]*CA2))
        C2=C2+dt*(0.08*(c2-C2)/tc2)
    
        #STN effs
        vef[:,i]=Ve+dt*(1/Cm*(-Il2e-Ik2e-Ina2e-It2e-Ica2e-Iahp2e+Idbs[i]-Ic2-Igesne+Iappstn-Ippnsne-Icorsne-Isncsne)) #currently STN-DBS 
        N2e=N2e+dt*(0.75*(n2e-N2e)/tn2e)
        H2e=H2e+dt*(0.75*(h2e-H2e)/th2e)
        R2e=R2e+dt*(0.2*(r2e-R2e)/tr2e)   
        CA2e=CA2e+dt*(3.75*1e-5*(-Ica2e-It2e-kca[1]*CA2e))
        C2e=C2e+dt*(0.08*(c2e-C2e)/tc2e)
        #for second-order alpha-synapse
        a=np.where(np.logical_and(vef[:,i-1]<-10, vef[:,i]>-10))[0]
        u=np.zeros(n) 
        u[a]=gpeak/(tau*np.exp(-1))/dt 
        S2=S2+dt*Z2
        zdot=u-2/tau*Z2-1/(tau**2)*S2
        Z2=Z2+dt*zdot
        #for stn-ppn/snr/snc/prf synapse
        S5=S5+dt*(A[1]*(1-S5)*Hinf(Ve-the[1])-B[1]*S5)    
    
        #GPe
        vge[:,i]=V3+dt*(1/Cm*(-Il3-Ik3-Ina3-It3-Ica3-Iahp3+Iappgpe-Isnge-Istrge-Igege-Ippnge-Isncge-Icorge))
        N3=N3+dt*(0.1*(n3-N3)/tn3) #misspelled in So paper
        H3=H3+dt*(0.05*(h3-H3)/th3) #misspelled in So paper
        R3=R3+dt*(1*(r3-R3)/tr3) #misspelled in So paper
        CA3=CA3+dt*(1*1e-4*(-Ica3-It3-kca[2]*CA3))
        #ge-sn/str/snr/snc synapse
        S3=S3+dt*(A[2]*(1-S3)*Hinf(V3-the[2])-B[2]*S3)
    
        #GPi
        vgi[:,i]=V4+dt*(1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4+Iappgpi-Isngi-Igegi-Ippngi-Istrgi-Isncgi))
        N4=N4+dt*(0.1*(n4-N4)/tn4) #misspelled in So paper
        H4=H4+dt*(0.05*(h4-H4)/th4) #misspelled in So paper
        R4=R4+dt*(1*(r4-R4)/tr4) #misspelled in So paper
        CA4=CA4+dt*(1*1e-4*(-Ica4-It4-kca[2]*CA4))
        #for second-order alpha-synapse
        a=np.where(np.logical_and(vgi[:,i-1]<-10, vgi[:,i]>-10))[0]
        u=np.zeros(n) 
        u[a]=gpeak1/(tau*np.exp(-1))/dt 
        S4=S4+dt*Z4 
        zdot=u-2/tau*Z4-1/(tau**2)*S4
        Z4=Z4+dt*zdot
        #for gpi-ppn/snc synapse
        S6=S6+dt*(A[3]*(1-S6)*Hinf(V4-the[3])-B[3]*S6)
    
        #PPN
        vppn[:,i]=V5+dt*(1/Cm*(-Inal5-Ikl5-Ina5-Ik5-It5-Ihyp5-Inap5+Iappppn-Icort-Isnppn-Igippn-Isnrppn-Iprfppn-Icnfppn))
        H5=H5+dt*((h5-H5)/th5)
        M5=M5+dt*((m5-M5)/tm5)
        Mk5=Mk5+dt*((mk5-Mk5)/tmk5)
        Mh5=Mh5+dt*((mh5-Mh5)/tmh5)
        Mt5=Mt5+dt*((mt5-Mt5)/tmt5)
        Ht5=Ht5+dt*((ht5-Ht5)/tht5)
        Hnap5=Hnap5+dt*((hnap5-Hnap5)/thnap5) #alike rubin
        #for ppn-stn/gpi/prf/cnf/snr/gpe/snc synapse
        S7=S7+dt*(A[4]*(1-S7)*Hinf(V5-the[4])-B[4]*S7)
    
        #SNr
        #vsnr[:,i]=V6+dt*(1/Cm*(-Il6-Ik6-Ina6-It6-Ica6-Iahp6-Isnsnr-Istrsnr-Iprfsnr-Icnfsnr-Ippnsnr-Igesnr-Isncsnr+Idbs[i]+Iappsnr)) 
        #with effs
        ch1 = V6+dt*(1/Cm*(-Il6-Ik6-Ina6-It6-Ica6-Iahp6-Isnsnr-Istrsnr-Iprfsnr-Icnfsnr-Ippnsnr-Igesnr-Isncsnr+Iappsnr-Ic3))
        if len(ch1[np.logical_and(ch1>-10, Idbs[i]>10)])!=0:
            ch1[np.logical_and(ch1>-10, Idbs[i]>10)]=ch1[np.logical_and(ch1>-10, Idbs[i]>10)]
        ch1[np.logical_not(np.logical_and(ch1>-10, Idbs[i]>10))]=\
            ch1[np.logical_not(np.logical_and(ch1>-10, Idbs[i]>10))]+dt*(1/Cm*(0.1*Idbs[i])) #why 0.01???
        vsnr[:,i]=ch1
        #vsnr[:,i]=V6+dt*(1/Cm*(-Il6-Ik6-Ina6-It6-Ica6-Iahp6+Iappsnr-Isnsnr-Istrsnr-Iprfsnr-Icnfsnr-Ippnsnr-Igesnr-Isncsnr-Ic3))
        N6=N6+dt*(0.1*(n6-N6)/tn6) #misspelled in So paper
        H6=H6+dt*(0.05*(h6-H6)/th6) #misspelled in So paper
        R6=R6+dt*(1*(r6-R6)/tr6) #misspelled in So paper
        CA6=CA6+dt*(1*1e-4*(-Ica6-It6-kca[2]*CA6))

        #SNr effs
        vsnre[:,i]=Vsnre+dt*(1/Cm*(-Il6e-Ik6e-Ina6e-It6e-Ica6e-Iahp6e-Isnsnre-Istrsnre-Iprfsnre-Icnfsnre-Ippnsnre-Igesnre-Isncsnre+Iappsnr-Ic4))
        #vsnre[:,i]=Vsnre+dt*(1/Cm*(-Il6e-Ik6e-Ina6e-It6e-Ica6e-Iahp6e-Ic4+Iappsnr-Isnsnre-Istrsnre-Iprfsnre-Icnfsnre-Ippnsnre-Igesnre-Isncsnre))#
        N6e=N6e+dt*(0.1*(n6e-N6e)/tn6e) #misspelled in So paper
        H6e=H6e+dt*(0.05*(h6e-H6e)/th6e) #misspelled in So paper 
        R6e=R6e+dt*(1*(r6e-R6e)/tr6e) #misspelled in So paper
        CA6e=CA6e+dt*(1*1e-4*(-Ica6e-It6e-kca[2]*CA6e))
        S14=S14+dt*(A[5]*(1-S14)*Hinf(Vsnre-the[5])-B[5]*S14) #snr-ppn/cnf/prf/str/snc synapse
    
        #Striatum D2
        vstr[:,i]=V7+(dt/Cm)*(-Ina7-Ik7-Il7-Im7-Igaba7-Icorstr+Iappstr-Igestr-Isnrstr-Isncstr)
        m7=m7+dt*(str_alpham(V7)*(1-m7)-str_betam(V7)*m7)
        h7=h7+dt*(str_alphah(V7)*(1-h7)-str_betah(V7)*h7)
        n7=n7+dt*(str_alphan(V7)*(1-n7)-str_betan(V7)*n7)
        p7=p7+dt*(str_alphap(V7)*(1-p7)-str_betap(V7)*p7)
        S1c=S1c+dt*((str_Ggaba(V7)*(1-S1c))-(S1c/tau_i))
        #for str-gpe/gpi/str/snc synapse
        S10=S10+dt*(A[6]*(1-S10)*Hinf(V7-the[6])-B[6]*S10) 
    
        #PRF - HH with T-current taken from GPe
        vprf[:,i]=V8+dt*(1/Cm*(-Il8-Ik8-Ina8-It8+Iappprf-Icorprf-Ippnprf-Ilcprf-Icnfprf-Isnrprf-Istnprf))
        N8=N8+dt*(0.1*(n8-N8)/tn8) 
        H8=H8+dt*(0.05*(h8-H8)/th8) 
        R8=R8+dt*(1*(r8-R8)/tr8) 
        #for prf-lc/cnf/ppn/snr
        S19=S19+dt*(A[8]*(1-S19)*Hinf(V8-the[8])-B[8]*S19)
    
        #CNF alike GPi and SNr
        vcnf[:,i]=V9+dt*(1/Cm*(-Il9-Ina9-Ik9-It9-Ica9-Iahp9+Iappcnf-Isnrcnf-Iprfcnf-Ippncnf)) #-Igicnf
        # H9=H9+dt*((h9-H9)/th9)
        # M9=M9+dt*((m9-M9)/tm9)
        # Mk9=Mk9+dt*((mk9-Mk9)/tmk9)
        # Mh9=Mh9+dt*((mh9-Mh9)/tmh9) 
        # Mt9=Mt9+dt*((mt9-Mt9)/tmt9)      
        # Ht9=Ht9+dt*((ht9-Ht9)/tht9)
        # Hnap9=Hnap9+dt*((hnap9-Hnap9)/thnap9)
        N9=N9+dt*(0.1*(n9-N9)/tn9) #misspelled in So paper
        H9=H9+dt*(0.05*(h9-H9)/th9) #misspelled in So paper
        R9=R9+dt*(1*(r9-R9)/tr9) #misspelled in So paper
        CA9=CA9+dt*(1*1e-4*(-Ica9-It9-kca[2]*CA9))
        #for cnf-lc/prf/ppn/snr 
        S18=S18+dt*(A[7]*(1-S18)*Hinf(V9-the[7])-B[7]*S18)
    
        #LC Arnauds paper
        vlc[:,i]=V10+dt*(1/Cm*(-Ina10-Ik10-Il10+Iapplc-Icnflc-Iprflc))
        q=q+dt*((qinf-q)/tq)
        #lc-prf/snc synapse
        S20=S20+dt*(A[9]*(1-S20)*Hinf(V10-the[9])-B[9]*S20) 

        #SNc - alike GPe and SNr
        vsnc[:,i]=V11+dt*(1/Cm*(-Il11-Ik11-Ina11-It11-Ica11-Iahp11+Iappsnc-Ilcsnc-Ippnsnc-Isnrsnc-Isnsnc-Igisnc-Igesnc-Istrsnc-Icorsnc))
        N11=N11+dt*(0.1*(n11-N11)/tn11) #misspelled in So paper
        H11=H11+dt*(0.05*(h11-H11)/th11) #misspelled in So paper
        R11=R11+dt*(1*(r11-R11)/tr11) #misspelled in So paper
        CA11=CA11+dt*(1*1e-4*(-Ica11-It11-kca[2]*CA11))
        #for snc-snr/stn/gpi/gpe/str synapses
        S12=S12+dt*(A[10]*(1-S12)*Hinf(V11-the[10])-B[10]*S12) 

    # plt.figure()
    # plt.plot(Sc[0,:])
    # plt.plot(S8[0])

    return vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx

    
    
