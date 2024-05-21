# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:34:04 2019

@author: nayab, juliettevl
"""

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
import fit_TDSf as TDS
import h5py #Library to import the noise matrix
from collections import Counter
from pathlib import  Path as path_
import numdifftools as nd


import warnings
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!


# =============================================================================
j = 1j
c = 2.998e8

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
    if myrank==0:
        print("Error importing pyopt")
    
try:
    from pyOpt import SLSQP  ## Library for optimization
except:
    if myrank==0:
        print("Error importing pyopt SLSQP")


# =============================================================================
# classes we will use
# =============================================================================

class globalparameters:
   def __init__(self, t, freq,w):
      self.t = t
      self.freq = freq
      self.w = w

# =============================================================================

class inputdatafromfile:
   def __init__(self, path):
      self.timeAndPulse = np.loadtxt(path) ## We load the data of the measured pulse
      self.Pulseinit = self.timeAndPulse[:,1]
      self.Spulseinit = (np.fft.rfft((self.Pulseinit)))  ## We compute the spectrum of the measured pulse

# =============================================================================

class mydata:
   def __init__(self, pulse,refSpulse):
      self.pulse= pulse
      self.Spulse= np.fft.rfft((pulse))
      self.mytransferfunction = np.fft.rfft((pulse))/refSpulse
#      self.epsilon= dielcal(np.fft.rfft((pulse))/refSpulse,z,myglobalparameters) #pas possible a n couches
#      self.Spulse= fft_gpu((pulse))
#      self.mytransferfunction = fft_gpu((pulse))/refSpulse
      self.mynorm= np.linalg.norm(refSpulse)

# =============================================================================

class myfitdata:
    def __init__(self, layers, delay_guess = 0, leftover_guess = np.zeros(2)):
        self.mytransferfunction = layers.transferfunction(myglobalparameters.w,delay_guess,leftover_guess)
        self.pulse = self.calculedpulse(layers,delay_guess,leftover_guess)
        self.Spulse = (np.fft.rfft((self.calculedpulse(layers,delay_guess,leftover_guess))))
        self.epsilon = []
        optim_materials = []
        for layer in layers.layers:
            if layer.material.fit_material: #changer pour plrs fois meme mat
                optim_materials.append(layer.material)
        for mat in optim_materials:
            self.epsilon.append(mat.epsilon(myglobalparameters.w))
            
        # =============================================================================
    # function that returns the convolved pulse to the transfer function, it does it by different Drude model with one oscillator, n oscillators, etc
    # =============================================================================
    def calculedpulse(self,layers,delay_guess,leftover_guess):
        global myinputdata,myreferencedata, myglobalparameters
        Z = layers.transferfunction(myglobalparameters.w,delay_guess=delay_guess,leftover_guess=leftover_guess)
        Spectrumtot=Z*myreferencedata.Spulseinit
        Pdata=(np.fft.irfft((np.array(Spectrumtot)), n = len(myreferencedata.Pulseinit)))
        return Pdata

# =============================================================================
class Callback_bfgs(object):
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_bfgs_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))
            
class Callback_slsqp(object):
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, convergence=0):
        self.nit += 1
        with open('algo_slsqp_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))


class Callback_annealing(object):
    def __init__(self):
        self.nit = 0
        
    def __call__(self, par, f, context):
        self.nit += 1
        with open('algo_dualannealing_out.txt', 'a+') as filehandle:
            filehandle.write('\n iteration number %d ; error %s ; parameters %s \r\n' % (self.nit, monerreur(par), par))

           



# =============================================================================
def errorchoice(i):
    global myinputdata, input_reduced, normalisedWeight, normalisedNoise, nsample, spulsenorm, pulsenorm, mode, normalisednoisemat
    if mode == "basic":
        if i == 0: # constant weight. Maybe we we should only keep this one in the normal resolution.
            def monerreur(x):
                Z = fit_transfer_function(x)
                fit_pulse = np.fft.irfft(Z*myreferencedata.Spulseinit)
                erreur=np.linalg.norm((fit_pulse-myinputdata.pulse))/pulsenorm
                return erreur
        elif i == 1: # custom weighting
            def monerreur(x):
                Z = fit_transfer_function(x)
                fit_pulse = np.fft.irfft(Z*myreferencedata.Spulseinit, n = len(myreferencedata.Pulseinit))
                erreur=np.linalg.norm((fit_pulse-myinputdata.pulse)*normalisedWeight)/pulsenorm#spulsenorm#/myinputdata.mynorm
                return erreur
        elif i == 3:
            def monerreur(x):
                Z = fit_transfer_function(x)
                fit_pulse = np.fft.irfft(Z*myreferencedata.Spulseinit, n = len(myreferencedata.Pulseinit))
                Rtls = np.dot(normalisednoisemat,(myinputdata.pulse-fit_pulse))
                #Rtls = np.diag(normalisednoisemat)*(myinputdata.pulse-fit_pulse)
                #Stls = np.dot(np.transpose(Rtls),Rtls)
                erreur = np.sqrt(np.dot(np.transpose(Rtls),Rtls))/pulsenorm #virer pulsenorm ?
                return erreur
        else:
            def monerreur(x):
                Z = fit_transfer_function(x)
                fit_pulse = np.fft.irfft(Z*myreferencedata.Spulseinit, n = len(myreferencedata.Pulseinit))
                erreur=np.linalg.norm((fit_pulse-myinputdata.pulse)/normalisedNoise)/pulsenorm#spulsenorm#/myinputdata.mynorm
                return erreur
    elif mode == "superresolution":
        if i == 0: # constant weight:
            def monerreur(x):
                Z = fit_transfer_function(x)
                Spectrumtot=Z*myreferencedata.Spulseinit
                pulse_theo=(np.fft.irfft((np.array(Spectrumtot)), n = len(myreferencedata.Pulseinit))) # calcul from calculedpulse. In fact it is the same calcul as in the basic mode for i!=0
                pulse_theo_reduced = pulse_theo[:nsample]
                
                erreur=np.linalg.norm(input_reduced-pulse_theo_reduced)/pulsenorm
                return erreur
        elif i == 1: # custom weighting
            def monerreur(x):
                Z = fit_transfer_function(x)
                Spectrumtot=Z*myreferencedata.Spulseinit
                pulse_theo=np.fft.irfft((np.array(Spectrumtot)),n = len(myreferencedata.Pulseinit))

                pulse_theo_reduced = pulse_theo[:nsample]
		
                erreur=np.linalg.norm((input_reduced-pulse_theo_reduced)*normalisedWeight)/pulsenorm
                return erreur
                return erreur
    return monerreur

#def guesstest():
#    global myinputdata, normalisedWeight, normalisedNoise, nsample, spulsenorm, pulsenorm, mode, normalisednoisemat
#    minval = np.array(list(minDict.values()))
#    maxval = np.array(list(maxDict.values()))
#    xguess = np.array(list(myVariablesDictionary.values()))
#    print('xguess')
#    print(xguess)
#    x = (xguess-minval)/(maxval-minval)
#    monerreur = errorchoice(error_index)
#    f = monerreur(x)
#    return f

# =============================================================================

def fit_transfer_function(x):
    global mylayerlist, myinterfacelist, myglobalparameters, weightferf, position_optim_thickness,nb_param, position_optim_material, fitDelay, delaymax_guess, delay_limit, delayfixed, fitLeftover, leftcoef_guess, leftcoef_limit, leftfixed
    x1 = x*(maxval-minval)+minval
    delay_guess = 0
    leftover_guess = np.zeros(2)
    if fitLeftover:
        count=-1
        for i in range (0,len(leftcoef_guess)):                    
            count = count+1
            leftover_guess[i] = x1[-len(leftcoef_guess)+count]
    if fitDelay:
        if fitLeftover:
            delay_guess = x1[-len(leftcoef_guess)-1]
        else:
            delay_guess = x1[-1]
    for i in position_optim_material:
        mylayerlist[i].material.change_param(x1[0:nb_param],myvariables)
    for i in position_optim_interface:
        myinterfacelist[i].change_param(x1[0:nb_param],myvariables)
    for i, pos in enumerate(position_optim_thickness):
        mylayerlist[pos].thickness = x1[nb_param+i]
    mylayers = TDS.Layers(mylayerlist, myinterfacelist)
    return mylayers.transferfunction(myglobalparameters.w, delay_guess, leftover_guess)

# =============================================================================
def errorchoice_pyOpt(i):
    def objfunc(x):  ## Function used in the Optimization function from pyOpt. For more details see http://www.pyopt.org/quickguide/quickguide.html
        monerreur = errorchoice(i)
        f = monerreur(x)
        fail = 0
        return f, 1, fail
    return objfunc


# =============================================================================

def optimALPSO(opt_prob, swarmsize, maxiter,algo,out_opt_full_info_filename):
    if algo == 2:
        alpso_none = ALPSO(pll_type='SPM')
    else:
        alpso_none = ALPSO()
    alpso_none.setOption('fileout',1)
    alpso_none.setOption('filename',out_opt_full_info_filename)
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
    
def optimSLSQP(opt_prob,maxiter,swarmsize):
    slsqp_none = SLSQP()
    slsqp_none.setOption('IPRINT',1)
    slsqp_none.setOption('IFILE',out_opt_full_info_filename)
    slsqp_none.setOption('MAXIT',maxiter)
    slsqp_none.setOption('IOUT',15) 
    slsqp_none.setOption('ACC',1e-20)
    return(slsqp_none(opt_prob))

def optimSLSQPpar(opt_prob,maxiter,swarmsize): # arecopierdansdoublet
          
    slsqp_none = SLSQP() # arecopierdansdoublet

    slsqp_none.setOption('IPRINT',1)
    slsqp_none.setOption('IFILE',out_opt_full_info_filename)
    slsqp_none.setOption('MAXIT',maxiter)
    slsqp_none.setOption('IOUT',12) 
    slsqp_none.setOption('ACC',1e-24)
    return(slsqp_none(opt_prob,sens_mode='pgc')) 

            

# =============================================================================
# Change that if the GPU is needed
# =============================================================================
def fft_gpu(y):
#    global using_gpu
#    if using_gpu==1:
#        ygpu = cp.array(y)
#        outgpu=cp.fft.rfft(ygpu)  # implied host->device
#        out=outgpu.get()
#    else:
    out = np.fft.rfft(y)
    return(out)

def ifft_gpu(y):
#    global using_gpu
#    if using_gpu==1:  ## Only works if the number of elements before doing the fft is pair
#        ygpu = cp.array(y)
#        outgpu=cp.fft.irfft(ygpu)  # implied host->device
#        out=outgpu.get()                            
#    else:
    out = np.fft.irfft(y)
    return(out)

# =============================================================================
# We load the model choices
# =============================================================================
f=open(os.path.join("temp",'temp_file_1_ini.bin'),'rb')
[pathwithoutsample,pathwithsample, freqWindow, timeWindow, fitDelay, delaymax_guess, delay_limit, delayfixed, mode, fitLeftover, leftcoef_guess, leftcoef_limit, leftfixed]=pickle.load(f)
#[myinputdata, myreferencedata, myglobalparameters, nsample, delaymax, mode] = pickle.load(f)
f.close()

f=open(os.path.join("temp",'temp_file_1.bin'),'rb')
[myvariables, epsilonTarget]=pickle.load(f)
f.close()

f=open(os.path.join("temp",'temp_file_4.bin'),'rb')
[out_dir,time_domain_filename,frequency_domain_filename,out_opt_filename]=pickle.load(f)
f.close()

f=open(os.path.join("temp",'temp_file_5.bin'),'rb')
[algo,swarmsize,maxiter,error_index,error_file]=pickle.load(f)
f.close()

#using_gpu = 0



# Load fields data

frequency_domain_filename = os.path.join(out_dir,frequency_domain_filename)
time_domain_filename = os.path.join(out_dir,time_domain_filename)
out_opt_full_info_filename=os.path.join(out_dir,'{0}_full_info.out'.format(out_opt_filename.split('.')[0]))
out_opt_filename = os.path.join(out_dir,out_opt_filename)
out_rtls_filename = os.path.join(out_dir,out_opt_filename+'_rtls')
    
datawithsample=np.loadtxt(pathwithsample)    ## We load the signal of the measured pulse with sample

myreferencedata=inputdatafromfile(pathwithoutsample) # champs
    
myglobalparameters=globalparameters   # t freq w
myglobalparameters.t=myreferencedata.timeAndPulse[:,0]*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
nsample=len(myglobalparameters.t)
dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate
myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)        ## We create a list with the frequencies for the spectrum
myglobalparameters.w=myglobalparameters.freq*2*np.pi
    
myinputdata=mydata(datawithsample[:,1],myreferencedata.Spulseinit)    ## We create a variable containing the data related to the measured pulse with sample

if mode == "superresolution":
    frep=99.991499600e6 # repetition frequency of the pulse laser used in the tds measurments in Hz, 99
    nsampleZP=np.round(1/(frep*dt)) #number of time sample betwen two pulses. IT has to be noted that it could be better to have an integer number there then the rounding does not change much
    nsamplenotreal=nsampleZP.astype(int)
    myglobalparameters.t=np.arange(nsampleZP)*dt  # 0001 #
    myglobalparameters.freq = np.fft.rfftfreq(nsamplenotreal, dt)
    myglobalparameters.w = 2*np.pi*myglobalparameters.freq
    
    myreferencedata.Pulseinit=np.pad(myreferencedata.timeAndPulse[:,1],(0,nsamplenotreal-nsample),'constant',constant_values=(0))
    myreferencedata.Spulseinit=(fft_gpu((myreferencedata.Pulseinit)))    # fft computed with GPU
    
    myinputdata=mydata(np.pad(datawithsample[:,1],(0,nsamplenotreal-nsample),'constant',constant_values=(0)),myreferencedata.Spulseinit)
#    monepsilon=dielcal(fft_gpu((np.pad(datawithsample[:,1],(0,nsamplenotreal-nsample),'constant',constant_values=(0))))/myreferencedata.Spulseinit,z,myglobalparameters)
#print(myglobalparameters.w)
#print(len(myglobalparameters.w))
# Filter data
myreferencedata.Spulseinit = myreferencedata.Spulseinit*freqWindow
myinputdata.Spulse         = myinputdata.Spulse        *freqWindow
myreferencedata.Pulseinit  = np.fft.irfft(myreferencedata.Spulseinit, n = len(myreferencedata.Pulseinit))
myinputdata.pulse          = np.fft.irfft(myinputdata.Spulse, n = len(myinputdata.pulse))

myreferencedata.Pulseinit = myreferencedata.Pulseinit*timeWindow
myreferencedata.Spulseinit = (np.fft.rfft((myreferencedata.Pulseinit)))

myinputdata=mydata(myinputdata.pulse,myreferencedata.Spulseinit)
#for superresolution
input_reduced = myinputdata.pulse[:nsample] #input_reduced norm is equal to pulsenorm

# error weight
#print(error_file)
pulsenorm = np.linalg.norm(myinputdata.pulse)
if error_file is not None:
    if error_index == 1:
        weight = np.loadtxt(error_file)
        try:
            if len(weight[0]) == 2: #in case there is time
                weight = weight[:,1]
        except:
            pass
        spulsenorm = np.linalg.norm(myinputdata.Spulse)
        pulsenorm = np.linalg.norm(myinputdata.pulse)
        weightnorm = np.linalg.norm(weight)/np.linalg.norm(np.ones(nsample))
        normalisedWeight = weight/weightnorm
    elif error_index == 2:
        noise = np.loadtxt(error_file)
        try:
            if len(noise[0]) == 2: #in case there is time
                noise = noise[:,1]
        except:
            pass
        spulsenorm = np.linalg.norm(myinputdata.Spulse)
        pulsenorm = np.linalg.norm(myinputdata.pulse)
        noisenorm = np.linalg.norm(noise)/np.linalg.norm(np.ones(nsample))
        normalisedNoise = noise/noisenorm
    elif error_index == 3:
        noisedata = h5py.File(error_file, 'r')
        name = list(noisedata.keys())[0]
        noisematrix = np.array(list(noisedata[name]))
        #noisematrix = np.diagflat(np.ones(np.size(myinputdata.pulse)))
        #noisematnorm = np.linalg.norm(noisematrix)/np.linalg.norm(np.ones(nsample))
        noisematnorm = np.linalg.norm(noisematrix)
        #print(noisematnorm)
        normalisednoisemat = noisematrix/noisematnorm
        pulsenorm = np.linalg.norm(myinputdata.pulse)
#        try:
#            if len(noise[0]) == 2: #in case there is time
#                noise = noise[:,1]
#        except:
#            pass
#        spulsenorm = np.linalg.norm(myinputdata.Spulse)
#        pulsenorm = np.linalg.norm(myinputdata.pulse)
#        noisenorm = np.linalg.norm(noise)/np.linalg.norm(np.ones(nsample))
#        normalisedNoise = noise/noisenorm
monerreur = errorchoice(error_index)
objfunc = errorchoice_pyOpt(error_index)

# =============================================================================

# We load the parameters values
f=open(os.path.join("temp",'temp_file_2.bin'),'rb')
[position_optim_thickness, position_optim_material, position_optim_interface, mylayers, mylayerlist, myinterfacelist, mesparam]=pickle.load(f)
f.close()


# =============================================================================
# change to optimize several materials/metamaterials. make list of myvariables/mesparam

nb_param = len(myvariables)

myVariablesDictionary = dict(zip(myvariables,mesparam[:,0]))
minDict = dict(zip(myvariables,mesparam[:,1]))
maxDict = dict(zip(myvariables,mesparam[:,2]))

myglobalparameters
myinputdata
myreferencedata
mylayers
totVariablesName = myvariables

# =============================================================================

for pos in position_optim_thickness:
    layer = mylayerlist[pos]
    z = layer.thickness
    deltaz = layer.uncertainty
    myVariablesDictionary['thickness_{}'.format(pos)]=z
    minDict['thickness_{}'.format(pos)] = z*(1-deltaz)
    maxDict['thickness_{}'.format(pos)] = z*(1+deltaz)
    totVariablesName = np.append(myvariables,'thickness_{}'.format(pos))
if fitDelay:
    if not delayfixed:
        myVariablesDictionary['delay']=delaymax_guess
        minDict['delay'] = -delay_limit
        maxDict['delay'] =  delay_limit
        totVariablesName = np.append(totVariablesName,'delay')
    else:
        myVariablesDictionary['delay']=delaymax_guess
        minDict['delay'] = delaymax_guess
        maxDict['delay'] = delaymax_guess+delaymax_guess/1e6 if delaymax_guess[i]!=0 else 1e-50
        totVariablesName = np.append(totVariablesName,'delay')
if fitLeftover:
    tab=[]
    for i in range (0,len(leftcoef_guess)):                    
        if not leftfixed[i]:
            myVariablesDictionary['leftover '+str(i)]=leftcoef_guess[i]#leftcoef[count-1]
            minDict['leftover '+str(i)] = -leftcoef_limit[i]
            maxDict['leftover '+str(i)] = leftcoef_limit[i]
            tab = np.append(tab,'leftover '+str(i))
        else:
            myVariablesDictionary['leftover '+str(i)]=leftcoef_guess[i]#leftcoef[count-1]
            minDict['leftover '+str(i)] = leftcoef_guess[i]
            maxDict['leftover '+str(i)] = leftcoef_guess[i]+leftcoef_guess[i]/1e6 if leftcoef_guess[i]!=0 else 1e-50
            tab = np.append(tab,'leftover '+str(i))
    totVariablesName = np.append(totVariablesName,tab)
## We take into account the thicknesses and delay as optimization parameters
# so we put the values and their uncertainty in the corresponding lists


"""parameters for the optimization algorithm"""


#=============================================================================#
# Instantiate Optimization Problem
#=============================================================================#

# Normalisation
minval = np.array(list(minDict.values()))
maxval = np.array(list(maxDict.values()))
guess = np.array(list(myVariablesDictionary.values()))
print('guess')
print(guess)

#if guess>=0:
x0=np.array((guess-minval)/(maxval-minval))
#else:
#    x0=-(guess-minval)/(maxval-minval)
print('x0')
print(x0)
print('errorguess')
print(monerreur(x0))
lb=np.zeros(len(guess))
up=np.ones(len(guess))


## Optimization dans le cas PyOpt
if algo in [1,2,3,4]:
    opt_prob = Optimization('Dielectric modeling based on TDS pulse fitting',objfunc)
    for nom,varvalue in myVariablesDictionary.items():
        #if varvalue>=0:
        opt_prob.addVar(nom,'c',lower = 0,upper = 1,
                    value = (varvalue-minDict.get(nom))/(maxDict.get(nom)-minDict.get(nom)) #normalisation
                    )
        #else:
        #    opt_prob.addVar(nom,'c',lower = 0,upper = 1,
        #                value = -(varvalue-minDict.get(nom))/(maxDict.get(nom)-minDict.get(nom)) #normalisation
         #               )    
    opt_prob.addObj('f')
    #opt_prob.addCon('g1','i') #possibility to add constraints
    #opt_prob.addCon('g2','i')


# =============================================================================
# solving the problem with the function in scipy.optimize
# =============================================================================


if  algo==0: ## xopt is a list we the drudeinput's parameters that minimize 'monerreur', fopt is a list with the optimals objective values
    start = time.process_time()
    xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) ## 'monerreur' function that we want to minimize, 'lb' and 'up' bounds of the problem
    elapsed_time = time.process_time()-start
    print("Time taken by the optimization:",elapsed_time)
    
if algo == 5:
    start = time.process_time()
    cback=Callback_bfgs()
    res = optimize.minimize(monerreur,x0,method='L-BFGS-B',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter})
    elapsed_time = time.process_time()-start
    hess = nd.hessian(monerreur,step=1e-6)(res.x)
    xopt = res.x
    fopt = res.fun
    print(res.message,"\nTime taken by the optimization:",elapsed_time)
    print(f"hess_bfgs : {res['hess_inv']}")
    print(f"hess_nd : {hess}")
    
if algo == 6:
    start = time.process_time()
    cback=Callback_slsqp()
    print('error x0 algo')
    print(monerreur(x0))
    res = optimize.minimize(monerreur,x0,method='SLSQP',bounds=list(zip(lb, up)),callback=cback,options={'maxiter':maxiter, 'ftol': 1e-20})
    elapsed_time = time.process_time()-start
    xopt = res.x
    fopt = res.fun
    print(res.message,"\nTime taken by the optimization:",elapsed_time)
    
if algo==7:
    start = time.process_time()
    cback=Callback_annealing()
    res = optimize.dual_annealing(monerreur, bounds=list(zip(lb, up)),callback=cback,maxiter=maxiter)
    elapsed_time = time.process_time()-start
    xopt = res.x
    fopt = res.fun
    print(res.message,"\nTime taken by the optimization:",elapsed_time)



# =============================================================================
# solving the problem with pyOpt
# =============================================================================


if  (algo==1)|(algo == 2):
    start = time.process_time()
    [fopt, xopt, inform] = optimALPSO(opt_prob, swarmsize, maxiter,algo,out_opt_full_info_filename)
    elapsed_time = time.process_time()-start
    print(inform,"\nTime taken by the optimization:",elapsed_time)
    
if algo ==3:
        try:
            start = time.process_time()
            [fopt, xopt, inform] = optimSLSQP(opt_prob,maxiter,swarmsize)
            elapsed_time = time.process_time()-start
            print(inform,"\nTime taken by the optimization:",elapsed_time)
        except Exception as e:
            print(e)

if algo ==4:
        try:
            start = time.process_time()
            [fopt, xopt, inform] = optimSLSQPpar(opt_prob,maxiter,swarmsize)
            elapsed_time = time.process_time()-start
            print(inform,"\nTime taken by the optimization:",elapsed_time)
        except Exception as e:
            print(e)
            

# =============================================================================
    
if myrank == 0:
    xopt = xopt*(maxval-minval)+minval
    text_result=[]
    text_result.append('The best error was: \t{}\n'.format(fopt))
    text_result.append('the best parameters were: \t{}\n'.format(xopt))
    
    # =========================================================================
    delay_guess = 0
    leftover_guess = np.zeros(2)
    if fitLeftover:
        count=-1
        for i in range (0,len(leftcoef_guess)):                    
            count=count+1
            leftover_guess[i] = xopt[-len(leftcoef_guess)+count]
    if fitDelay:
        if fitLeftover:
            delay_guess = xopt[-len(leftcoef_guess)-1]
        else:
            delay_guess = xopt[-1]
    for i in position_optim_material:
        mylayerlist[i].material.change_param(xopt,myvariables)
    for i in position_optim_interface:
        myinterfacelist[i].change_param(xopt,myvariables)
    for i, pos in enumerate(position_optim_thickness):
        mylayerlist[pos].thickness = xopt[nb_param+i]
    mylayers = TDS.Layers(mylayerlist, myinterfacelist)
    myfitteddata=myfitdata(mylayers,delay_guess=delay_guess,leftover_guess=leftover_guess)


    # =========================================================================
    # saving the results
    # =========================================================================


    outputtime=np.column_stack((myglobalparameters.t,myfitteddata.pulse))

    text_result.append("\n Please cite this paper in any communication about any use of fit@tds :")
    text_result.append("\n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters")
    text_result.append("\n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas")
    text_result.append("\n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2")
    text_result.append("\n DOI: 10.1109/TTHZ.2018.2889227 \n")

    result_optimization=[xopt,text_result]
    f=open(os.path.join("temp",'temp_file_3.bin'),'wb')
    pickle.dump(result_optimization,f,pickle.HIGHEST_PROTOCOL)
    f.close()

    ## Save the data obtained via this program
    # Save optimization parameters
    Rtls_opt=0
    if error_index == 3:
        Rtls_opt = np.dot(noisematrix,(myinputdata.pulse-myfitteddata.pulse))
        Stls = np.dot(np.transpose(Rtls_opt),Rtls_opt)
        Qaic = Stls + 2*np.size(xopt)
        print('Akaike criterion')
        print(Qaic)
    outputoptim = fopt
    if error_index == 3:
        outputoptim = np.append(outputoptim,Qaic)
        np.savetxt(out_rtls_filename,Rtls_opt)
    outputoptim = np.append(outputoptim,xopt)
    
    out_opt_h5 = h5py.File(out_opt_filename+'.h5', 'w')
    dset = out_opt_h5.create_dataset("output", (len(outputoptim),),dtype='float64')
    dset[:]=outputoptim

    out_opt_h5.close()
    
    np.savetxt(out_opt_filename,outputoptim) #np.linalg.norm(myinputdata.pulse)
    
    # Save time domain results
    np.savetxt(time_domain_filename,outputtime,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n time \t E-field")
    
    #print(np.real(epsilonTarget), np.imag(epsilonTarget), np.real(np.sqrt(epsilonTarget)),np.imag(np.sqrt(epsilonTarget)))
#    if (epsilonTarget is not None)&(len(mylayerlist) == 1):
    try:
        outputfreq=abs(np.column_stack((myglobalparameters.freq,myfitteddata.Spulse,np.real(myfitteddata.epsilon[0]),np.imag(myfitteddata.epsilon[0]),
                                        np.real(np.sqrt(myfitteddata.epsilon[0])),np.imag(np.sqrt(myfitteddata.epsilon[0])),np.real(epsilonTarget) ,
                                        np.imag(epsilonTarget), np.real(np.sqrt(epsilonTarget)),np.imag(np.sqrt(epsilonTarget)) )))
#    else:
    except:
        outputfreq=abs(np.column_stack((myglobalparameters.freq,myfitteddata.Spulse,np.real(myfitteddata.epsilon[0]),np.imag(myfitteddata.epsilon[0]),
                                        np.real(np.sqrt(myfitteddata.epsilon[0])),np.imag(np.sqrt(myfitteddata.epsilon[0])) )))

    # Save frequency domain results
    np.savetxt(frequency_domain_filename,outputfreq,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n Freq \t E-field \t real part of fitted epsilon \t imaginary part of fitted epsilon \t real part of fitted n \t imaginary part of fitted n \t real part of initial epsilon \t imaginary part of initial epsilon \t real part of initial n\t imaginary part of initial n")


    # =========================================================================
    # History and convergence
    # =========================================================================
    out_opt_full_info_filename_dir = path_(out_opt_full_info_filename).parent
    out_opt_full_info_filename = path_(out_opt_full_info_filename).stem
    out_opt_full_info_filename = path_(out_opt_full_info_filename_dir).joinpath(out_opt_full_info_filename)
    
    if  (algo==1)|(algo == 2):
        # f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r')
        f = open(f"{out_opt_full_info_filename}_print.out",'r')
    
        # Find maxiter
        line = f.readline()
        while (line !=''):
            line = f.readline()
            lineSplit = line.split()
            if len(lineSplit)>2:
                if (lineSplit[0:3]==['NUMBER','OF','ITERATIONS:']):
                    maxiter = int(lineSplit[3])
            
        f.close()
        # f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r') # Go back to the beginning
        f = open(f"{out_opt_full_info_filename}_print.out",'r')# Go back to the beginning
        
        # To find number of parameters:
        j = 0
        while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<50):
            j = j+1
        P = [(f.readline())[4:]]
        
        while (f.readline()!= 'BEST POSITION:\n') & (j<100):
            j =  j+1
        line = (f.readline())
        line = (line.split())
        nLines = 0
        while(len(line)>0):#(line[0][0] == 'P'):
            P.extend(line[2::3])
            line = (f.readline())
            line = (line.split())
            nLines = nLines+1
        bestPositions = np.zeros((maxiter,len(P)))
        bestPositions[0]=P
        
        
        for i in range(1,maxiter):
            j = 0
            while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<100):
                j = j+1 # to avoid infinite loop
                # One could use pass instead
            P = [(f.readline())[4:]]
            while (f.readline()!= 'BEST POSITION:\n') & (j<200):
                j =  j+1
            for nLine in range(nLines):
                line = (f.readline())
                line = (line.split())
                P.extend(line[2::3])
            bestPositions[i]=P
        f.close()
        
        # Write and save file
        historyHeaderFile = os.path.join(out_dir,"convergence.txt")
        historyHeader = 'objective function value'
        for name in myVariablesDictionary:
            historyHeader = '{}{}\t'.format(historyHeader, name)
        np.savetxt(historyHeaderFile, bestPositions, header = historyHeader)
