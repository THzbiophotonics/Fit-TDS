#!/usr/bin/env python

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math
from pyswarm import pso
import random
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import Tkinter as tk
import tkFileDialog 
from finalplot import plotall
from finalplot import plotinput
from epsillon3 import dielcal

import warnings
warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!

###############################################################################
###############################################################################
j = 1j
c = 2.998e8
###############################################################################
# =============================================================================
# External Python modules (serves for algo #3)
# =============================================================================
try: # this can be commented if mpi4py was not installed successfully 
    from mpi4py import MPI # this can be commented if mpi4py was not installed successfully 
    comm = MPI.COMM_WORLD # this can be commented if mpi4py was not installed successfully 
    myrank = comm.Get_rank() # this can be commented if mpi4py was not installed successfully 
except: # this can be commented if mpi4py was not installed successfully 
    raise ImportError('mpi4py is required for parallelization') # this can be commented if mpi4py was not installed successfully 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#end
# =============================================================================
# Extension modules
# =============================================================================
#from pyOpt import *
from pyOpt import Optimization
from pyOpt import ALPSO
from pyOpt import SLSQP
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
   def __init__(self, pulse,inputSpulse,z):
      self.z= z
      self.pulse= pulse
      self.Spulse= np.fft.rfft((pulse))
      self.mytransferfunction = np.fft.rfft((pulse))/inputSpulse
      self.epsilon= dielcal(np.fft.rfft((pulse))/inputSpulse,z,myglobalparameters)
      self.mynorm= np.linalg.norm(inputSpulse)
###############################################################################
class myfitdata:
   def __init__(self, xopt):
      self.mytransferfunction = transferfunction(xopt) 
      self.pulse= calculedpulse(xopt)
      self.Spulse= (np.fft.rfft((calculedpulse(xopt))))
      self.epsilon= Drude(xopt)
###############################################################################
# sanitised_input function from https://stackoverflow.com/questions/23294658/asking-the-user-for-input-until-they-give-a-valid-response
###############################################################################
def sanitised_input(prompt, type_=None, min_=None, max_=None, range_=None):
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min_ must be less than or equal to max_.")
    while True:
        ui = raw_input(prompt)# replace raw_inut byi= input for python 3
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else:
                    print(template.format(" or ".join((", ".join(map(str,range_[:-1])),str(range_[-1])))))
        else:
            return ui
##############################################################################
#  Here one can put an additional refractive index model like Debye for ex.
##############################################################################
def Drude(drudeinput):
	global  myglobalparameters,n,zvariable,mymodelstruct,isdrude
	interm=0
	if zvariable==1:
		interm=interm+1	
	if mymodelstruct==2:
		interm=interm+5
	
	eps_inf =drudeinput[0+interm]
	eps =eps_inf*np.ones(len(myglobalparameters.w))
	
	if isdrude==1:	
		#metaldrude
		wp=drudeinput[1+interm]
		gamma =drudeinput[2+interm]
		eps =eps- wp**2/(1E0+myglobalparameters.w**2-j*gamma* myglobalparameters.w)		
		interm= interm+2
	for i in range(0,n):
		chi=drudeinput[i*3+1+interm]
		w0=drudeinput[i*3+2+interm]*2*np.pi
		gamma =drudeinput[i*3+3+interm]*2*np.pi 
		eps =eps+ chi*w0**2/(w0**2+j*gamma* myglobalparameters.w- myglobalparameters.w**2)
	return eps
##############################################################################
#defining the transfer function
##############################################################################
def transferfunction( drudeinput ):
	global  myglobalparameters,n,zvariable,mymodelstruct
	interm=0 
#first question: is thickness a parameter for optimization?
	if zvariable==1:
		interm=interm+1
		thickn=drudeinput[0]
	else:
		thickn=z
#second question: will we use the TDCMT model or not? 
#ref for TDCMT: https://ieeexplore.ieee.org/document/784592/ 
	if mymodelstruct==2:
		w0=drudeinput[0+interm]
		tau0=drudeinput[interm+1]
		tau1=drudeinput[interm+2]
		tau2=drudeinput[interm+3]
		deltatheta=drudeinput[interm+4]
		taue=2/((1/tau1)+(1/tau2))
#		drudeinputbis=np.zeros(len(drudeinput)-5-interm)
		interm=interm+5	

	ref_index=np.sqrt(Drude(np.array(drudeinput)))

#caculation of all the transmission and reflection coefficients
	t12=2/(1+ref_index)
	t21=2*ref_index/(1+ref_index)
	r22=(ref_index-1)/(1+ref_index)
	r22b=r22

	if mymodelstruct==2:
		deltaw=myglobalparameters.w-w0
		interm1=1/((j*deltaw)+(1/taue)+(1/tau0))
		t12=t12-(1/tau1)*interm1
		r22b=r22b-np.exp(-j*deltatheta)*interm1/np.sqrt(tau1*tau2)

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
	global myinputdata, myinputdata,weight
	erreur=np.linalg.norm((myinputdata.mytransferfunction-transferfunction(drudeinput))*myinputdatafromfile.Spulseinit)/myinputdata.mynorm 
	return erreur
##############################################################################
def objfunc(x):
	f = monerreur(x)
	#g = 1[0.0]*2
	#g[0] =  1
	#g[1] = 1
	fail = 0
	return f, 1,fail
# =============================================================================
# retrieval of the data
# =============================================================================
if myrank == 0:
    root =tk.Tk()
    print("\nPlease choose the input file for the pulse without sample\n")
    pathwithoutsample=tkFileDialog.askopenfilename(parent=root)
    
    print("\nPlease choose the input file for the pulse with sample\n")
    pathwithsample=tkFileDialog.askopenfilename(parent=root)
    root.destroy()
    mesdata=np.loadtxt(pathwithsample)
    
    myinputdatafromfile=inputdatafromfile
    myglobalparameters=globalparameters
    myinputdatafromfile.PulseInittotal=np.loadtxt(pathwithoutsample)
    
    myglobalparameters.t=myinputdatafromfile.PulseInittotal[:,0]*1e-12 #this assumes input files are in ps
    nsample=len(myglobalparameters.t)
    myinputdatafromfile.Pulseinit=myinputdatafromfile.PulseInittotal[:,1]
    dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)
    myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)
    myglobalparameters.w=myglobalparameters.freq*2*np.pi
    myinputdatafromfile.Spulseinit=(np.fft.rfft((myinputdatafromfile.Pulseinit)))
    
    z= sanitised_input("\nEnter a value for the thickness in meters [m]:", float, 0)
    deltaz= sanitised_input("\nEnter a value for the thickness uncertainty in %:", float, 0)/100
    
    myinputdata=mydata(mesdata[:,1],myinputdatafromfile.Spulseinit,z)
    monepsilon=dielcal(np.fft.rfft((mesdata[:,1]))/myinputdatafromfile.Spulseinit,z,myglobalparameters)
    
    ##############################################################################
    #parameters for the algorithm 
    ##############################################################################
    """parameters for the algorithm"""
    swarmsize=1000
    maxiter=20
    algo=2 
    #algorithm 1 stands for NumPy optimize swarm particle, 
    #algorithm 2 - for PyOpt swarm particle ALPSO without parallelization // 3 - ALPSO with parallelization //
    
    # =============================================================================
    # calculating the delay to infer the index
    # =============================================================================
    deltaT=myglobalparameters.t[np.argmax(myinputdata.pulse)]-myglobalparameters.t[np.argmax(myinputdatafromfile.Pulseinit)] #retard entre les deux max
    print("############################################################################################")
    print("############################################################################################")
    print("Delay between the two maxima of the pulses:")
    print("delta T="+ str(deltaT))
    print("n="+ str(1+deltaT*c/z)) #indice qui en derive
    print("epsillon="+ str(np.square(1+deltaT*c/z))) #indice qui en derive
    print("############################################################################################")
    print("############################################################################################")
    deltaTTT=myglobalparameters.t[np.argmin(myinputdata.pulse)]-myglobalparameters.t[np.argmin(myinputdatafromfile.Pulseinit)] #retard entre les deux max
    print("Delay between the two minima of the pulses:")
    print("delta T="+ str(deltaTTT))
    print("n="+ str(1+deltaTTT*c/z)) #indice qui en derive
    print("epsillon="+ str(np.square(1+deltaTTT*c/z))) #indice qui en derive
    print("############################################################################################")
    print("############################################################################################")
    deltaTT=np.sum(np.square(myinputdata.pulse)*myglobalparameters.t)/np.sum(np.square(myinputdata.pulse))-np.sum(np.square		(myinputdatafromfile.Pulseinit)*myglobalparameters.t)/np.sum(np.square(myinputdatafromfile.Pulseinit))   #retard entre les deux barycentre, attention pour que ca foncionne il faut que le rapport signal bruit soit le meme dans les deux cas !!
    print("############################################################################################")
    print("############################################################################################")
    print("Delay between the two energy barycenter of the pulses\n (beware that noise brings it to the middle for each one):")
    print("delta T="+ str(deltaTT))
    print("n="+ str(deltaTT*c/z)) #indice qui en derive
    print("epsillon="+ str(np.square(deltaTT*c/z))) #indice qui en derive
    print("############################################################################################")
    print("############################################################################################")
    ###############################################################################
    ###############################################################################
    #Ploting of the input. 
    #The goal will be then to get proper initial values for parameters.
    ###############################################################################
    ###############################################################################
    plotinput(monepsilon,myinputdata,myinputdatafromfile,z,myglobalparameters)
    ###############################################################################
    ###############################################################################
    mymodelmat=1
    #we may add here the choice between Drude Lorentz and Debye
    #we may add if one would want to put a scattering term 
    
    mymodelstruct= sanitised_input("\nWich model do you whant to use for the photonic structure ? \n\t 1 - Transmission Fabry-Perot\n\t 2 - Transmission Fabry-Perot with a resonator (TDCMT) \n\t\t", int, 1,2)
    zvariable= sanitised_input("Is the thickness a variable for the fit ? \n\t 0 - NO \n\t 1 - Yes \n\t\t", int, 0,1)
    
    
    ###############################################################################
    #Input questions
    ###############################################################################
    #short description of drude model
    print("\n Drude model depicts the permitivity Epsillon as Eps =Eps_0- Omega_p^2/(Omega^2-j*gamma*omega) ")
    
    isdrude=sanitised_input("Do you want to have a Drude term in the model ? \n\t 0 - NO \n\t 1 - Yes \n\t\t", int, 0,1)
    
    #short description of Lorentz model
    print("\n Lorentz model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon*Omega_0^2]/[Omega_0^2+j*gamma*Omega-Omega^2] ")
    
    
    print("")
    
    n= sanitised_input("\nEnter a value for the number of Lorentz oscillators in the model: \t", int, 0)
    
    if mymodelstruct==2:
        for i in range(0,1):
            myvariables=["Omega resonator/metasurface_"+str(i), "Tau 0 resonator/metasurface_" +str(i),"Tau 1 resonator/metasurface_"+str(i),"Tau 2 resonator/metasurface_"+str(i),"delta Theta resonator/metasurface_"+str(i) ]
            myunits=["Radian / s", "s" ,"s","s","Radian"]
            mydescription=["central angular frequency of the mode of the resonator # "+str(i), "Absorption life time of the mode of the resonator # " +str(i),"Forward coupling lifetime of the mode of the resonator # "+str(i),"Backward coupling lifetuime of the mode of the resonator # "+str(i),"Phase between Forward and backward coupling for the resontator # "+str(i) ]
    else:
        myvariables=[]
        myunits=[]
        mydescription=[]
    
    myvariables=myvariables+["epsillon_inf"]
    myunits=myunits+["Usual permitivity unit without dimension (square of a refractive index)"]
    mydescription=mydescription+["Permitivity at very high frequency frequency"]
    if isdrude==1:
        myvariables=myvariables+["Omega_p"]+["gamma"]
        myunits=myunits+["Radian/s"]+["Radian/s"]
        mydescription=mydescription+["Drude Model Plasma frequency : [ (N * (q^2))  /  (Epsillon_0  * m_e) ]"]+["Drude damping damping rate"]
    for i in range(0,n):
        myvariables=myvariables+["Delta_Epsillon_"+str(i), "1/(2pi)*Omega0_" +str(i),"1/(2pi)*Gamma_"+str(i) ] 
        myunits=myunits+["Usual permitivity unit without dimension (square of a refractive index)", "Hz","Hz" ]
        mydescription=mydescription+["Oscillator strentgh of the mode # ", "Frequency of the mode # " +str(i),"Linewidth of the mode # "+str(i) ] 
    drudeinput=np.ones(len(myvariables))
    lb=np.ones(len(myvariables))
    up=np.ones(len(myvariables))
    compteur=0
    
    prompt=1
    if prompt:
        compteur=0
    
        for i in myvariables:
            while True:
                drudeinput[compteur]= sanitised_input("\nEnter a value for "+ str(i)+" in SI unit: \t", float, 0)
                lb[compteur]= sanitised_input("Enter a value for the minimum value of "+ str(i)+": \t", float, 0)
                up[compteur]= sanitised_input("Enter a value for the maximum value of "+ str(i)+": \t", float, 0)
                if  up[compteur]<=lb[compteur]:
                        print("The maximum of "+ str(i)+" must be strictly higher than its minimum")
                        continue
                elif  up[compteur]<drudeinput[compteur]:
                        print("The maximum of "+ str(i)+" must be higher than its value")
                        continue
                elif  lb[compteur]>drudeinput[compteur]:
                        print("The minimum of "+ str(i)+" must be lower than its value")
                        continue
                else:
                        break
        
            compteur=compteur+1
    else:
        root2 =tk.Tk()
        print("\nPlease choose the file where all the parameters for the model are\n")
        pathparam=tkFileDialog.askopenfilename(parent=root2)
        mesparam=np.loadtxt(pathparam)
        drudeinput=np.array(mesparam[:, 0])
        lb=np.array(mesparam[:, 1])
        up=np.array(mesparam[:, 2])
        root2.destroy()
        root2.mainloop()
    
    
    # =============================================================================
    #interm=0  #this is not only in case of a simulated sample
    #
    #if zvariable==1:
    #	drudeinput=np.append([z],drudeinput)
    #	lb=np.append([z*(1-deltaz)],lb)
    #	up=np.append([z*(1+deltaz)],up)
    #	interm=interm+1                #this is not only in case of simulated sample
    #if mymodelstruct==2:              #if one use resonator tdcmt 
    #	interm=interm+5                #this is not only in case of simulated sample
    #if  mymodelmat==2:                #if one use a variable thickness
    #	drudeinput=np.append([Scatfreq],drudeinput)
    #	interm=interm+1                #this is not only in case of simulated sample
    
    # =============================================================================
    # preparation of the input for the algorithm
    # =============================================================================
    print("############################################################################################")
    print("############################################################################################")
    print("begining the calculation please wait ...")
    print("############################################################################################")
    print("############################################################################################")
    
    # Instantiate Optimization Problem 
    inputsize=np.size(lb)   
inputsize= comm.bcast(inputsize, root=0) 
if myrank != 0:
    lb=np.empty(inputsize, dtype='f')
    up=np.empty(inputsize, dtype='f')
    drudeinput=np.empty(inputsize, dtype='f')
comm.Bcast(lb, root=0)
comm.Bcast(drudeinput, root=0)
comm.Bcast(up, root=0)

 
if algo>1:
    interm2=0
    opt_prob = Optimization('Dielectric modeling based on TDS pulse fitting',objfunc)
    if zvariable==1:
        opt_prob.addVar('thickness','c',lower=lb[0],upper=up[0],value=drudeinput[0])
        interm2=interm2+1
    if mymodelstruct==2:       #in case of TDCMT
        opt_prob.addVar('w0 tdcmt','c',lower=lb[0+interm2],upper=up[0+interm2],value=drudeinput[0+interm2])
        opt_prob.addVar('tau0','c',lower=lb[1+interm2],upper=up[1+interm2],value=drudeinput[1+interm2])
        opt_prob.addVar('tau1','c',lower=lb[2+interm2],upper=up[2+interm2],value=drudeinput[2+interm2])
        opt_prob.addVar('tau2','c',lower=lb[3+interm2],upper=up[3+interm2],value=drudeinput[3+interm2])
        opt_prob.addVar('delta theta','c',lower=lb[4+interm2],upper=up[4+interm2],value=drudeinput[4+interm2])
        interm2=interm2+5

    opt_prob.addVar('eps inf','c',lower=lb[0+interm2],upper=up[0+interm2],value=drudeinput[0+interm2])
    if isdrude==1:
        opt_prob.addVar('omega p','c',lower=lb[1+interm2],upper=up[1+interm2],value=drudeinput[1+interm2])
        opt_prob.addVar('gamma','c',lower=lb[2+interm2],upper=up[2+interm2],value=drudeinput[2+interm2])
        interm2=interm2+2

    for i in range(0,n):
        opt_prob.addVar('chi'+str(i),'c',lower=lb[1+interm2+3*i],upper=up[1+interm2+3*i],value=drudeinput[1+interm2+3*i])#pour drude
        opt_prob.addVar('w'+str(i),'c',lower=lb[2+interm2+3*i],upper=up[2+interm2+3*i],value=drudeinput[2+interm2+3*i])#pour drude
        opt_prob.addVar('gamma'+str(i),'c',lower=lb[3+interm2+3*i],upper=up[3+interm2+3*i],value=drudeinput[3+interm2+3*i])#pour drude
    
    opt_prob.addObj('f')
    #opt_prob.addCon('g1','i') #possibility to add constraintes
    #opt_prob.addCon('g2','i')
        
    ##############################################################################
    ##############################################################################
    # solving the problem with the function in scipy.optimize
    ##############################################################################
    ##############################################################################
    if  algo==1:
        xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) 
    ##############################################################################
    ##############################################################################
    # solving the problem with yopt
    ##############################################################################
    ##############################################################################
swarmsize= comm.bcast(swarmsize, root=0)
maxiter= comm.bcast(maxiter, root=0)

if  algo==2:
# Solve Problem (No-Parallelization)
	alpso_none = ALPSO()#pll_type='SPM')
	alpso_none.setOption('fileout',1)
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
	alpso_none.setOption('HoodSize',swarmsize/100)
	[fopt, xopt, inform]=alpso_none(opt_prob)
	if myrank == 0:# this can be commented if mpi4py was not installed successfully 
		print(opt_prob.solution(0))# this can be commented if mpi4py was not installed successfully 
	#end
##############################################################################
##############################################################################
if myrank == 0:
    print("the best error was: \t" + str(fopt)+"\n")
    print("the best parameters were: \t" + str(xopt)+"\n")
    ##############################################################################
    myfitteddata=myfitdata(xopt)
    ##############################################################################
    ##############################################################################
    # final plot of the results
    ##############################################################################
    ##############################################################################
    plotall(myinputdata,myinputdatafromfile,myfitteddata,monepsilon,myglobalparameters)
    ##############################################################################
    #saving the results
    ##############################################################################
    ##############################################################################
    
    outputtime=np.column_stack((myglobalparameters.t,myfitteddata.pulse))
    
    
    print("\n Please cite this paper in any communication about any use of fit@tds :")
    print("\n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters")
    print("\n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, and Jean-Francois Lampin")
    print("\n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2")
    print("\n DOI: 10.1109/TTHZ.2018.2889227 \n")
    
    print("\n Please choose the file name and path to save the fit results in time domain\n")
    root3=tk.Tk()
    pathoutputime=tkFileDialog.asksaveasfilename()
    fileoutputtime=open(pathoutputime,'w')
    np.savetxt(fileoutputtime,outputtime,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, and Jean-Francois Lampin \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n time \t E-field")
    fileoutputtime.close()
    
    outputfreq=abs(np.column_stack((myglobalparameters.freq,myfitteddata.Spulse,np.real(myfitteddata.epsilon),np.imag(myfitteddata.epsilon), np.real(np.sqrt(myfitteddata.epsilon)),np.imag(np.sqrt(myfitteddata.epsilon)),np.real(monepsilon) ,np.imag(monepsilon), np.real(np.sqrt(monepsilon)),np.imag(np.sqrt(monepsilon)) )))
    print("\n Please choose the file name and path to save the fit results in frequency domain\n")
    pathoutpufreq=tkFileDialog.asksaveasfilename()
    fileoutputfreq=open(pathoutpufreq,'w')
    np.savetxt(fileoutputfreq,outputfreq,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, and Jean-Francois Lampin \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n Freq \t E-field \t real part of fitted epsilon \t imaginary part of fitted epsilon \t real part of fitted n \t imaginary part of fitted n \t real part of initial epsilon \t imaginary part of initial epsilon \t real part of initial n\t imaginary part of initial n")
    fileoutputfreq.close()
    root3.destroy()
    root3.mainloop()
