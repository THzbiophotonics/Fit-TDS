#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math
import pickle
from pyswarm import pso   ## Library for optimization
import random
import numpy as np   ## Library to simplify the linear algebra calculations
import scipy.optimize as optimize  ## Library for optimization
import matplotlib.pyplot as plt ## Library for plotting results
from scipy.optimize import curve_fit ## Library for optimization
from epsillon3 import dielcal ## Library for resolving the inverse problem in our case (see the assumptions necessary to use this library)


import warnings
warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!



###############################################################################
###############################################################################
j = 1j
c = 2.998e8
###############################################################################
# =============================================================================
# External Python modules (serves for optimization algo #3)
# =============================================================================
## Parallelization that requieres mpi4py to be installed, if mpi4py was not installed successfully comment frome line 32 to line 40 (included)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
    print('mpi4py is required for parallelization')
    myrank=0


#end
# =============================================================================
# Extension modules
# =============================================================================

try:
    from pyOpt import Optimization   ## Library for optimization
    from pyOpt import ALPSO  ## Library for optimization
except:
    print("Error importing pyopt")
#from pyOpt import SLSQP  ## Library for optimization

# =============================================================================
# classes we will use
# =============================================================================

class globalparameters:
   def __init__(self, t, freq,w):
      self.t = t
      self.freq = freq
      self.w = w

###############################################################################

class inputdatafromfile:
# maybe one can separate intput data from files with variable input
   def __init__(self, PulseInittotal, Pulseinit,Spulseinit,mytransferfunction,mypulse,mySpulse,mynorm,myeps):
      self.PulseInittotal = PulseInittotal
      self.Pulseinit = Pulseinit
      self.Spulseinit = Spulseinit

###############################################################################

class mydata:
   def __init__(self, pulse,inputSpulse,z,myglobalparameters):
      self.z= z
      self.pulse= pulse
      self.Spulse= np.fft.rfft((pulse))
      self.mytransferfunction = np.fft.rfft((pulse))/inputSpulse
      self.epsilon= dielcal(np.fft.rfft((pulse))/inputSpulse,z,myglobalparameters)
#      self.Spulse= fft_gpu((pulse))
#      self.mytransferfunction = fft_gpu((pulse))/inputSpulse
#      self.epsilon= dielcal(fft_gpu((pulse))/inputSpulse,z,myglobalparameters)
      self.mynorm= np.linalg.norm(inputSpulse)

###############################################################################

class myfitdata:
   def __init__(self, xopt):
      global  myglobalparameters,myinputdata,myinputdatafromfile, z, deltaz,pathwithoutsample,pathwithsample, monepsilon, myvariables, myunits, mydescription, algo, mymodelstruct, zvariable, isdrude,scattering, n,nDebye,swarmsize,niter
      f=open(os.path.join("temp",'temp_file_1.bin'),'rb')
      [z, deltaz,pathwithoutsample,pathwithsample, monepsilon, myvariables, myunits,
       mydescription, algo, mymodelstruct, zvariable, isdrude,scattering, n,nDebye,swarmsize,niter]=pickle.load(f)
      f.close()

      myinputdatafromfile=inputdatafromfile
      myglobalparameters=globalparameters
      mesdata=np.loadtxt(pathwithsample)    ## We load the signal of the measured pulse with sample
      myinputdatafromfile.PulseInittotal=np.loadtxt(pathwithoutsample) ## We load the data of the measured reference pulse
      myglobalparameters.t=myinputdatafromfile.PulseInittotal[:,0]*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
      nsample=len(myglobalparameters.t)
      myinputdatafromfile.Pulseinit=myinputdatafromfile.PulseInittotal[:,1]
      dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate
      myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)        ## We create a list with the frequencies for the spectrum
      myglobalparameters.w=myglobalparameters.freq*2*np.pi
      myinputdatafromfile.Spulseinit=(np.fft.rfft((myinputdatafromfile.Pulseinit)))  ## We compute the spectrum of the measured reference pulse
      myinputdata=mydata(mesdata[:,1],myinputdatafromfile.Spulseinit,z,myglobalparameters)    ## We create a variable containing the data related to the measured pulse with sample
      myinputdata=mydata(mesdata[:,1],myinputdatafromfile.Spulseinit,z,myglobalparameters)
      self.mytransferfunction = transferfunction(xopt)
      self.pulse= calculedpulse(xopt)
      self.Spulse= (np.fft.rfft((calculedpulse(xopt))))
      self.epsilon= Drude(xopt)
      self.epsilon_scat = Drude(xopt)

##############################################################################
#  Here one can put an additional refractive index model like Debye for ex.
##############################################################################

def Drude(drudeinput):
    global  myglobalparameters,n,nDebye,zvariable,mymodelstruct,isdrude, scattering
    interm=0
    if zvariable==0:
        interm=interm+1
    if mymodelstruct==1:
        interm=interm+5

    if scattering ==0:
        beta = drudeinput[0 + interm]
        Scat_freq_min = drudeinput[1+interm]
        Scat_freq_max = drudeinput[2+interm]
        interm=interm+3

    eps_inf =drudeinput[0+interm]
    eps =eps_inf*np.ones(len(myglobalparameters.w))

    if isdrude==0:
        wp=drudeinput[1+interm]
        gamma =drudeinput[2+interm]
        eps =eps- wp**2/(1E0+myglobalparameters.w**2-j*gamma* myglobalparameters.w)
        interm= interm+2

    if scattering==0:
        omega=np.where(myglobalparameters.w<Scat_freq_max*2*np.pi,myglobalparameters.w,1e-299)
        omega=np.where(omega>Scat_freq_min*2*np.pi,omega,1e-299)
        alpha = beta*(omega/(2*np.pi*1e12))**3
        n_diff = - j*alpha
        eps = eps+n_diff**2+2*np.sqrt(eps)*n_diff

    for i in range(0,n):  ## Lorentz term
        chi=drudeinput[i*3+1+interm]
        w0=drudeinput[i*3+2+interm]*2*np.pi
        gamma =drudeinput[i*3+3+interm]*2*np.pi
        eps =eps+ chi*w0**2/(w0**2+j*gamma* myglobalparameters.w- myglobalparameters.w**2)

    for iDebye in range(0,nDebye):  ## Debye term
        chi=drudeinput[iDebye*2+1+interm]
        w0=drudeinput[iDebye*2+2+interm]*2*np.pi
        eps =eps+ chi/(1+j*myglobalparameters.w/w0)

    return eps

##############################################################################
#defining the transfer function
##############################################################################
def transferfunction(drudeinput):
    global  myglobalparameters,n,zvariable,mymodelstruct,scattering
    interm=0

    if zvariable==0:
        interm=interm+1
        thickn=drudeinput[0]
    else:
        thickn=z

    if mymodelstruct==1:
        w0=drudeinput[0+interm]
        tau0=drudeinput[interm+1]
        tau1=drudeinput[interm+2]
        tau2=drudeinput[interm+3]
        deltatheta=drudeinput[interm+4]
        taue=2/((1/tau1)+(1/tau2))
        interm=interm+5

    if scattering == 0:
        interm = interm + 3

    ref_index=np.sqrt(Drude(np.array(drudeinput)))

    #caculation of all the transmission and reflection coefficients
    t12=2/(1+ref_index)  ## Coefficients where 1 is the air and 2 the metamaterial at normal incidence
    t21=2*ref_index/(1+ref_index)
    r22=(ref_index-1)/(1+ref_index)
    r22b=r22

    if mymodelstruct==1:
        deltaw=myglobalparameters.w-w0
        interm1=1/((j*deltaw)+(1/taue)+(1/tau0))
        t12=t12-(1/tau1)*interm1
        r22b=r22b-np.exp(-j*deltatheta)*interm1/np.sqrt(tau1*tau2)

    # In case we have just two interfaces
    rr=r22*r22b
    tt=t12*t21

    propa=np.exp(-j*myglobalparameters.w*(ref_index)*thickn/c)
    propaair=np.exp(-j*myglobalparameters.w*thickn/c)
    FP=1/(1-rr*(propa**2))
    Z=tt*propa*FP/propaair
    return Z

##############################################################################
# function that returns the convolved pulse to the transfer function, it does it by different Drude model with one oscillator, n oscillators, etc
##############################################################################

def calculedpulse(drudeinput):
	global myinputdata,myinputdatafromfile, myglobalparameters,z
	Z= transferfunction( drudeinput )
	Spectrumtot=Z*myinputdatafromfile.Spulseinit
	Pdata=(np.fft.irfft((np.array(Spectrumtot))))
	return Pdata

##############################################################################

def monerreur(drudeinput):
	global myinputdata,weightferf
	erreur=np.linalg.norm((myinputdata.mytransferfunction-transferfunction(drudeinput))*myinputdatafromfile.Spulseinit)/myinputdata.mynorm
	return erreur

##############################################################################

def objfunc(x):  ## Function used in the Optimization function from pyOpt. For more details see http://www.pyopt.org/quickguide/quickguide.html
	f = monerreur(x)
	fail = 0
	return f, 1,fail

##############################################################################

def optimALPSO(opt_prob, swarmsize, maxiter,algo):
    if algo == 3:
        alpso_none = ALPSO(pll_type='SPM')
    else:
        alpso_none = ALPSO()
    alpso_none.setOption('fileout',1)
    alpso_none.setOption('filename',"test3.out")
    alpso_none.setOption('SwarmSize',swarmsize)
    alpso_none.setOption('maxInnerIter',6)
    alpso_none.setOption('etol',1e-5)
    alpso_none.setOption('rtol',1e-10)
    alpso_none.setOption('atol',1e-10)
    alpso_none.setOption('vcrazy',1e-4)
    alpso_none.setOption('dt',1e0)
    alpso_none.setOption('maxOuterIter',maxiter)
    alpso_none.setOption('stopCriteria',0)#Stopping Criteria Flag (0 - maxIters, 1 - convergence)
    alpso_none.setOption('printInnerIters',1)
    alpso_none.setOption('printOuterIters',1)
    alpso_none.setOption('HoodSize',int(swarmsize/100))
    return(alpso_none(opt_prob))
