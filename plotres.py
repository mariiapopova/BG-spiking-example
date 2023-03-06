# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:45:58 2021

@author: Maria
"""
from findfreq import *
from calculateEI import *
import matplotlib.pyplot as plt

def plotres(vsn, vge, vgi, vppn, vth, vstr, vsnr, vprf, vcnf, vlc, vsnc, vctx, t,timespike,tmax,Istim,dt):

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
    
    titleval=GN #variable for plotting title
    
    ##Plots membrane potential for one cell in each nucleus
    plt.figure()
    plt.subplot(3,5,1) #for 1st PPN neuron
    plt.plot(t,vppn[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80)
    plt.title('PPN, FR: %s Hz' %(int(round(fr4))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,2) 
    plt.plot(t,vth[0,:])
    plt.plot(t,Istim[0:int(tmax/dt)],'r'); #plot for 1st neuron both Istim and Vth
    plt.xlim(0, tmax)
    plt.ylim(-100, 20)
    plt.title('Thalamus, FR: %s Hz' %(int(round(fr5))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,3) #for 1st STN neuron
    plt.plot(t,vsn[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('STN, FR: %s Hz' %(int(round(fr1))))
    plt.ylabel('Vm (mV)') 
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,4) #for 1st GPe neuron
    plt.plot(t,vge[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('GPe, FR: %s Hz' %(int(round(fr2))))
    plt.ylabel('Vm (mV)') 
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,5) #for 1st GPi neuron
    plt.plot(t,vgi[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('GPi, FR: %s Hz' %(int(round(fr3))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,6) #for 1st striatum neuron
    plt.plot(t,vstr[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('Striatum, FR: %s Hz' %(int(round(fr6))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,7) #for 1st SNr neuron
    plt.plot(t,vsnr[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('SNr, FR: %s Hz' %(int(round(fr7))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,8) #for 1st PRF neuron
    plt.plot(t,vprf[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('PRF, FR: %s Hz' %(int(round(fr8))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,9) #for 1st CNF neuron
    plt.plot(t,vcnf[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('CNF, FR: %s Hz' %(int(round(fr9))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,10) #for 1st LC neuron
    plt.plot(t,vlc[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('LC, FR: %s Hz' %(int(round(fr10))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(3,5,11) #for 1st SNc neuron
    plt.plot(t,vsnc[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('SNc, FR: %s Hz' %(int(round(fr11))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')

    plt.subplot(3,5,12) #for 1st Ctx neuron
    plt.plot(t,vctx[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('Ctx, FR: %s Hz' %(int(round(fr12))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')

    plt.suptitle('Firing patterns in freezing of gait network \n Thalamic relay EI: %s' %(titleval))
    
    plt.show()
    
    return GN