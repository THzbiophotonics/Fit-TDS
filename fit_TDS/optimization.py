# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:34:04 2019

@author: nayab
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
#from fit_TDSf import *


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
    if myrank==0:
        print("Error importing pyopt")
#from pyOpt import SLSQP  ## Library for optimization



###############################################################################
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

    if isdrude==0:	 ## Drude term
        #metaldrude
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
def transferfunction( drudeinput ):
    global  myglobalparameters,n,zvariable,mymodelstruct, scattering
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

    if scattering ==0:
        interm = interm + 3

    ref_index=np.sqrt(Drude(np.array(drudeinput)))

    #caculation of all the transmission and reflection coefficients
    t12=2/(1+ref_index)
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
###############################################################################☺
# =============================================================================
# We load the model choices
f=open(os.path.join("temp",'temp_file_1.bin'),'rb')
[z, deltaz,pathwithoutsample,pathwithsample, monepsilon, myvariables, myunits,
 mydescription, algo, mymodelstruct, zvariable, isdrude, scattering, n, nDebye, swarmsize, maxiter]=pickle.load(f)
f.close()

f=open(os.path.join("temp",'temp_file_4.bin'),'rb')
[out_dir,time_domain_filename,frequency_domain_filename,out_opt_filename]=pickle.load(f)
f.close()

frequency_domain_filename = os.path.join(out_dir,frequency_domain_filename)
time_domain_filename = os.path.join(out_dir,time_domain_filename)
out_opt_full_info_filename=os.path.join(out_dir,'{0}_full_info.out'.format(out_opt_filename.split('.')[0]))
out_opt_filename = os.path.join(out_dir,out_opt_filename)


if myrank == 0:
    print(f'swarmsize = {swarmsize} \nmaxiter = {maxiter}')


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

# =============================================================================


# =============================================================================
# We load the parameters values
f=open(os.path.join("temp",'temp_file_2.bin'),'rb')
mesparam=pickle.load(f)
f.close()
# =============================================================================



nb_param = len(myvariables)

drudeinput=np.ones(nb_param)
lb=np.ones(nb_param)   ## Array with  the min value of the parameters of the model
up=np.ones(nb_param)   ## Array with  the max value of the parameters of the model

drudeinput=mesparam[:, 0]
lb=mesparam[:, 1]
up=mesparam[:, 2]

myglobalparameters
monepsilon
myinputdata
myinputdatafromfile


#    =============================================================================
interm=0  #this is not only in case of a simulated sample

if zvariable==0: ## We take into account the thickness as an optimization parameter so we put the value of the tickness and its uncertainty in the corresponding list
	drudeinput=np.append([z],drudeinput)
	lb=np.append([z*(1-deltaz)],lb)
	up=np.append([z*(1+deltaz)],up)
	interm=interm+1
if mymodelstruct==1:              #if one use resonator tdcmt
    interm=interm+5
if scattering == 0:
    interm = interm + 3

"""parameters for the optimization algorithm"""


#=============================================================================#
# Instantiate Optimization Problem
#=============================================================================#
## Optimization dans le cas PyOpt swarm particle ALPSO without parallelization (also works with parallelization)
if algo>0:
    interm2=0  ## Intermediate variable with a function similar to interm
    opt_prob = Optimization('Dielectric modeling based on TDS pulse fitting',objfunc)
    if zvariable==0:
        opt_prob.addVar('thickness','c',lower=lb[0],upper=up[0],value=drudeinput[0])
        interm2=interm2+1
    if mymodelstruct==1:       #in case of TDCMT
        opt_prob.addVar('w0 tdcmt','c',lower=lb[0+interm2],upper=up[0+interm2],value=drudeinput[0+interm2])
        opt_prob.addVar('tau0','c',lower=lb[1+interm2],upper=up[1+interm2],value=drudeinput[1+interm2])
        opt_prob.addVar('tau1','c',lower=lb[2+interm2],upper=up[2+interm2],value=drudeinput[2+interm2])
        opt_prob.addVar('tau2','c',lower=lb[3+interm2],upper=up[3+interm2],value=drudeinput[3+interm2])
        opt_prob.addVar('delta theta','c',lower=lb[4+interm2],upper=up[4+interm2],value=drudeinput[4+interm2])
        interm2=interm2+5
    if scattering == 0:
        opt_prob.addVar('beta','c',lower=lb[0+interm2],upper=up[0+interm2],value=drudeinput[0+interm2])
        opt_prob.addVar('Scat_freq_min','c',lower=lb[1+interm2],upper=up[1+interm2],value=drudeinput[1+interm2])
        opt_prob.addVar('Scat_freq_max','c',lower=lb[2+interm2],upper=up[2+interm2],value=drudeinput[2+interm2])
        interm2 = interm2 + 3


    opt_prob.addVar('eps inf','c',lower=lb[0+interm2],upper=up[0+interm2],value=drudeinput[0+interm2])
    if isdrude==0:
        opt_prob.addVar('omega p','c',lower=lb[1+interm2],upper=up[1+interm2],value=drudeinput[1+interm2])
        opt_prob.addVar('gamma','c',lower=lb[2+interm2],upper=up[2+interm2],value=drudeinput[2+interm2])
        interm2=interm2+2

    for i in range(0,n):
        opt_prob.addVar(f'chi_{i}','c',lower=lb[1+interm2+3*i],upper=up[1+interm2+3*i],value=drudeinput[1+interm2+3*i])#pour drude
        opt_prob.addVar(f'w_{i}','c',lower=lb[2+interm2+3*i],upper=up[2+interm2+3*i],value=drudeinput[2+interm2+3*i])#pour drude
        opt_prob.addVar(f'gamma_{i}','c',lower=lb[3+interm2+3*i],upper=up[3+interm2+3*i],value=drudeinput[3+interm2+3*i])#pour drude

    for iDebye in range(0,nDebye):
        opt_prob.addVar(f'chi_{iDebye}','c',lower=lb[1+interm2+3*iDebye],upper=up[1+interm2+3*iDebye],value=drudeinput[1+interm2+3*iDebye])#pour drude
        opt_prob.addVar(f'w_{iDebye}','c',lower=lb[2+interm2+3*iDebye],upper=up[2+interm2+3*iDebye],value=drudeinput[2+interm2+3*iDebye])#pour drude


    opt_prob.addObj('f')
    #opt_prob.addCon('g1','i') #possibility to add constraintes
    #opt_prob.addCon('g2','i')



##############################################################################
##############################################################################
# solving the problem with the function in scipy.optimize
##############################################################################
##############################################################################


if  algo==0: ## xopt is a list we the drudeinput's parameters that minimize 'monerreur', fopt is a list with the optimals objective values
    xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) ## 'monerreur' function that we want to minimize, 'lb' and 'up' bounds of the problem


##############################################################################
##############################################################################
# solving the problem with yopt
##############################################################################
##############################################################################

if  algo>=1:
    [fopt, xopt, inform] = optimALPSO(opt_prob, swarmsize, maxiter,algo,out_opt_full_info_filename)

##############################################################################
##############################################################################
if myrank == 0:
    text_result=[]
    text_result.append(f'The best error was: \t {fopt}\n')
    text_result.append(f'the best parameters were: \t{xopt}\n')
    ##############################################################################
    myfitteddata=myfitdata(xopt)
    ##############################################################################
    ##############################################################################
    # final plot of the results
    ##############################################################################
    ##############################################################################
#    plotall(myinputdata,myinputdatafromfile,myfitteddata,monepsilon,myglobalparameters)

    ##############################################################################
    #saving the results
    ##############################################################################
    ##############################################################################

    outputtime=np.column_stack((myglobalparameters.t,myfitteddata.pulse))

    text_result.append("\n Please cite this paper in any communication about any use of fit@tds :")
    text_result.append("\n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters")
    text_result.append("\n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas")
    text_result.append("\n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2")
    text_result.append("\n DOI: 10.1109/TTHZ.2018.2889227 \n")

#    result_optimization=[myfitdata.mytransferfunction,myfitdata.pulse,myfitdata.Spulse,myfitdata.epsilon,myfitdata.epsilon_scat,text_result]
    result_optimization=[xopt,text_result]
    f=open(os.path.join("temp",'temp_file_3.bin'),'wb')
    pickle.dump(result_optimization,f,pickle.HIGHEST_PROTOCOL)
    f.close()

    ## Save the data obtained via this program
    # Save optimization parameters
    np.savetxt(out_opt_filename,xopt)
    # Save time domain results
    np.savetxt(time_domain_filename,outputtime,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n time \t E-field")

    outputfreq=abs(np.column_stack((myglobalparameters.freq,myfitteddata.Spulse,np.real(myfitteddata.epsilon),np.imag(myfitteddata.epsilon),
                                    np.real(np.sqrt(myfitteddata.epsilon)),np.imag(np.sqrt(myfitteddata.epsilon)),np.real(monepsilon) ,
                                    np.imag(monepsilon), np.real(np.sqrt(monepsilon)),np.imag(np.sqrt(monepsilon)) )))

    # Save frequency domain results
    np.savetxt(frequency_domain_filename,outputfreq,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n Freq \t E-field \t real part of fitted epsilon \t imaginary part of fitted epsilon \t real part of fitted n \t imaginary part of fitted n \t real part of initial epsilon \t imaginary part of initial epsilon \t real part of initial n\t imaginary part of initial n")

    ###########################################################################
    ###########################################################################
    # History and convergence
    ###########################################################################
    ###########################################################################
    
    if  algo>0:
        f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r')
    
        # Recuperation de maxiter
        line = f.readline()
        while (line !=''):
            line = f.readline()
            lineSplit = line.split()
            if len(lineSplit)>2:
                if (lineSplit[0:3]==['NUMBER','OF','ITERATIONS:']):
                    maxiter = int(lineSplit[3])
            
        f.close()
        f = open("{0}_print.out".format(out_opt_full_info_filename.split('.')[0]),'r') #Pour revenir au debut mais il doit y avoir une commande pour ca
        
        #Tout n est pas dans la boucle car on ne connaît pas le nb de  parametres optimises a priori.
        j = 0
        while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<50):
            j = j+1
        P = [(f.readline())[4:]]
        
        while (f.readline()!= 'BEST POSITION:\n') & (j<100):
            j =  j+1
        line = (f.readline())
        line = (line.split())
        nLines = 0
        while(line[0][0] == 'P'):
            P.extend(line[2::3])
            line = (f.readline())
            line = (line.split())
            nLines = nLines+1
        bestPositions = np.zeros((maxiter,len(P)))
        bestPositions[0]=P
        
        
        for i in range(1,maxiter):
            j = 0
            while (f.readline()!= 'OBJECTIVE FUNCTION VALUE:\n') & (j<100):
                j = j+1 #ca me sert surtout a eviter de rentrer dans une boucle infinie en cas de pb.
                # On pourrait mettre un pass en soi.
            P = [(f.readline())[4:]]
            while (f.readline()!= 'BEST POSITION:\n') & (j<200):
                j =  j+1
            for nLine in range(nLines):
                line = (f.readline())
                line = (line.split())
                P.extend(line[2::3])
            bestPositions[i]=P
        f.close()
        
        # Ecriture du fichier
        historyHeaderFile = os.path.join(out_dir,"convergence.txt")
        historyHeader = 'objective function value'
        if zvariable==0:
            historyHeader = '{0} \t thickness'.format(historyHeader)
        if mymodelstruct==1:       #in case of TDCMT
            historyHeader = '{0} \t w0 tdcmt'.format(historyHeader)
            historyHeader = '{0} \t tau0'.format(historyHeader)
            historyHeader = '{0} \t tau1'.format(historyHeader)
            historyHeader = '{0} \t tau2'.format(historyHeader)
            historyHeader = '{0} \t delta theta'.format(historyHeader)
        if scattering == 0:
            historyHeader = '{0} \t beta'.format(historyHeader)
            historyHeader = '{0} \t Scat_freq_min'.format(historyHeader)
            historyHeader = '{0} \t Scat_freq_max'.format(historyHeader)
        historyHeader = '{0} \t Permittivity at high frequency'.format(historyHeader) #eps inf
        if isdrude==0:
            historyHeader = '{0} \t Drude:omega p'.format(historyHeader)
            historyHeader = '{0} \t gamma'.format(historyHeader)

        for i in range(0,n):
            historyHeader = '{0} \t Lorentz:oscillator strength of the mode {1}'.format(historyHeader,i) #chi
            historyHeader = '{0} \t w_{1}'.format(historyHeader,i) #Frequency of the mode
            historyHeader = '{0} \t Linewidth of the mode {1}'.format(historyHeader,i) #gamma
    
        for iDebye in range(0,nDebye):
            historyHeader = '{0} \t Debye:chi_{1}_Debye'.format(historyHeader,iDebye)
            historyHeader = '{0} \t w_{1}_Debye'.format(historyHeader,iDebye)

        np.savetxt(historyHeaderFile, bestPositions, header = historyHeader)
