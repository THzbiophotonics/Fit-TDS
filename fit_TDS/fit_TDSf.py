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
import scipy.signal as scp # Library for signal processing
import scipy.special as special
import matplotlib.pyplot as plt ## Library for plotting results
import traceback
from scipy.optimize import curve_fit ## Library for optimization
from epsillon3 import dielcal ## Library for resolving the inverse problem in our case (see the assumptions necessary to use this library)


import warnings
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
#we stricly advise to comment the above line as soon as you modify the code!



###############################################################################
###############################################################################
import fit_TDSm as Model
j = 1j
c = 2.998e8
h = 6.62607015E-34
k= 1.38064852E-23
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
    print(traceback.format_exc())
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
    print(traceback.format_exc())
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

# =============================================================================

class inputdatafromfile:
   def __init__(self, path):
      self.timeAndPulse = np.loadtxt(path) ## We load the data of the measured pulse
      self.Pulseinit = self.timeAndPulse[:,1]
      self.Spulseinit = (np.fft.rfft((self.Pulseinit)))  ## We compute the spectrum of the measured pulse

# =============================================================================

class mydata:
   def __init__(self, pulse,refSpulse):
      self.pulse = pulse # pulse with sample
      self.Spulse = np.fft.rfft((pulse)) # spectral field with sample
      self.mytransferfunction = np.fft.rfft((pulse))/refSpulse
#      self.Spulse= fft_gpu((pulse))
#      self.mytransferfunction = fft_gpu((pulse))/refSpulse
      self.mynorm= np.linalg.norm(refSpulse)

# =============================================================================

class myfitdata:
    def __init__(self, layers, delay_guess = 0, leftover_guess = np.zeros(2)):
      global  myglobalparameters, myinputdata, myreferencedata, pathwithoutsample, pathwithsample
      f=open(os.path.join("temp",'temp_file_1_ini.bin'),'rb')
#      [myinputdata, myreferencedata, myglobalparameters, nsample, delaymax, mode] = pickle.load(f)
      [pathwithoutsample,pathwithsample, freqWindow, timeWindow,  fitDelay, delaymax_guess, delay_limit, delayfixed, mode, fitLeftover, leftcoef_guess, leftcoef_limit, leftfixed]=pickle.load(f)
      f.close()
      
      datawithsample=np.loadtxt(pathwithsample)    ## We load the signal of the measured pulse with sample
      myreferencedata=inputdatafromfile(pathwithoutsample)
      
      myglobalparameters = globalparameters
      myglobalparameters.t = myreferencedata.timeAndPulse[:,0]*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
      nsample = len(myglobalparameters.t)
      dt = myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate
      myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)        ## We create a list with the frequencies for the spectrum
      myglobalparameters.w = myglobalparameters.freq*2*np.pi
      
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
      # Filter data
      myreferencedata.Spulseinit = myreferencedata.Spulseinit*freqWindow
      myinputdata.Spulse         = myinputdata.Spulse*freqWindow
      myreferencedata.Pulseinit  = np.fft.irfft(myreferencedata.Spulseinit, n = len(myreferencedata.Pulseinit))
      myinputdata.pulse          = np.fft.irfft(myinputdata.Spulse, n = len(myinputdata.pulse))
      
      myreferencedata.Pulseinit = myreferencedata.Pulseinit*timeWindow
      myreferencedata.Spulseinit = (np.fft.rfft((myreferencedata.Pulseinit)))
      
      self.mytransferfunction = layers.transferfunction(myglobalparameters.w,delay_guess,leftover_guess)
      self.pulse = self.calculedpulse(layers,delay_guess,leftover_guess)
      self.Spulse = (np.fft.rfft((self.pulse)))
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
        Pdata=np.fft.irfft((np.array(Spectrumtot)), n = len(myreferencedata.Pulseinit))
        return Pdata

# =============================================================================
# Classes for Materials and Layers
# =============================================================================

class Material:
    materialCounter=0 #allow to add a unique id to each Material, so it can find the right variables
    def __init__(self, name='', nbTerms = [], param=np.array([1]), down = None, up=None, file = None, fit_material = 0):
        self.id=Material.materialCounter
        Material.materialCounter+=1
        # get parameters from file
        if file:
            f = open(file)
            header = f.readline()
            f.close()
            header = header.split()
            choices = dict(zip(header[1::2],header[2::2]))
            name = choices.get('Name')
            nbTerms = []
            for model in Model.materialModels:
                nbTerms.append(int(choices.get(model.name)))
            param = np.loadtxt(file, dtype = np.float64)
            #print(param)
        # choices
        self.name = name
        if len(nbTerms) != 0:
            self.nbTerms = nbTerms
        else:
            self.nbTerms = [0]*len(Model.materialModels)
        if param.ndim == 0:
            self.param = np.array([param])
        else:
            self.param = param
        self.change_variables(self.nbTerms)
        self.up = up
        self.down = down
        self.fit_material = fit_material
        
        self.header = "Name {0}".format(self.name.split('.')[0])
        for i in range(len(Model.materialModels)):
            self.header+=" {0} {1}".format(Model.materialModels[i].name,self.nbTerms[i])
        
        if self.param is not None:    
            self.eps_inf = self.variableDictionary.get("epsilon_inf_{}".format(self.id))
    
    def change_param(self,param, variables, down=None,up=None):
        if param.ndim == 0:
            self.param = np.array([param])
        else:
            self.param = param
        if down is not None:
            self.down = down
        if up is not None:
            self.up = up
        if self.param is not None:
            try:
                self.variableDictionary = dict(zip(variables,self.param))
                self.eps_inf = self.variableDictionary.get("epsilon_inf_{}".format(self.id))
            except:
                print(traceback.format_exc())
                pass
        
    def change_variables(self, nbTerms):
        self.nbTerms = nbTerms
        self.myvariables=["epsilon_inf_{}".format(self.id)]
        self.myunits=["dimensionless"]
        if self.name != '':
            self.mydescriptions=["{} : Permittivity at very high frequency\n".format(self.name)]
        else:
            self.mydescriptions=["Permittivity at very high frequency\n"]
        for i in range(len(Model.materialModels)):
            for j in range(nbTerms[i]):
                for k in range(len(Model.materialModels[i].variableNames)):
                    self.myvariables.append("{0}_{1}_{2}_{3}".format(Model.materialModels[i].variableNames[k],i,j,self.id)) #Adds all the index necessary to make sure the name is unique in the list and avoid conflict
                    self.myunits.append(Model.materialModels[i].variableUnits[k])
                    if Model.materialModels[i].isCumulative:
                        if self.name != '':
                            self.mydescriptions.append("{0} : {1}{2}\n".format(self.name,Model.materialModels[i].variableDescriptions[k],j))
                        else:
                            self.mydescriptions.append("{0}{1}\n".format(Model.materialModels[i].variableDescriptions[k],j))
                    else:
                        if self.name != '':
                            self.mydescriptions.append("{0} : {1}\n".format(self.name,Model.materialModels[i].variableDescriptions[k]))
                        else:
                            self.mydescriptions.append("{0}\n".format(Model.materialModels[i].variableDescriptions[k]))
        try:
            self.variableDictionary = dict(zip(self.myvariables,self.param))
            #print(self.variableDictionary)
        except:
            print(traceback.format_exc())
            pass

    def variableNames(self):
        return self.myvariables
    def variableUnits(self):
        return self.myunits
    def variableDescriptions(self):
        return self.mydescriptions
   
    def epsilon(self, w):
        eps = (self.eps_inf+0*1j)*np.ones(len(w))
        
        for i in range(len(Model.materialModels)):
            for j in range(self.nbTerms[i]):
                paramList=[self.variableDictionary.get("{0}_{1}_{2}_{3}".format(variableName,i,j,self.id)) for variableName in Model.materialModels[i].variableNames]
                eps += Model.materialModels[i].epsilon(eps,w,paramList)
        return eps
    
# =============================================================================

class Layer:
    def __init__(self,thickness,material,uncertainty=0,fit_index=1, id_fp=1):
        self.thickness = thickness
        self.uncertainty = uncertainty/100
        self.fit_thickness = 1-fit_index # 1 (True) if Yes, 0 (False) is No
        self.material = material # object of type Material
        
    def update_ini(self, other_layer):
        self.thickness = other_layer.thickness
        self.uncertainty = other_layer.uncertainty
        self.fit_thickness = other_layer.fit_thickness
        # update material unless both are material to optimize
        if (self.material.fit_material == 0)|(other_layer.material.fit_material == 0): 
            self.material = other_layer.material
# =============================================================================

class Interface:
    interfaceCounter=0
    def __init__(self, name = '', isMetasurface = 0, nbTerms = [],param=np.array([1]), down = None, up=None, file = None, fit_metasurface = 0):
        self.id=Interface.interfaceCounter
        Interface.interfaceCounter+=1
        if file:                                                            #need to check if this need to change
            param = np.loadtxt(file, dtype = np.float64)
            f = open(file)
            header = f.readline()
            f.close
            header = header.split()
            choices = dict(zip(header[1::2],header[2::2]))
            name = choices.get("Name")
            isMetasurface = 1
        self.name = name
        self.isMetasurface = isMetasurface
        if len(nbTerms) != 0:
            self.nbTerms = nbTerms
        else:
            self.nbTerms = [0]*len(Model.interfaceModels)



        self.down = down
        self.up = up

        if param.ndim == 0:
            self.param = np.array([param])
        else:
            self.param = param
            
        self.change_variables(self.nbTerms)

        self.fit_metasurface = fit_metasurface
        
        
        self.header = "Name {0}".format(self.name.split('.')[0])
        for i in range(len(Model.interfaceModels)):
            self.header+=" {0} {1}".format(Model.interfaceModels[i].name,self.nbTerms[i])
        

    def change_variables(self, nbTerms):
        self.nbTerms = nbTerms
        self.myvariableNames=[]
        self.myunits=[]
        self.mydescriptions=[]
        for i in range(len(Model.interfaceModels)):
            for j in range(nbTerms[i]):
                for k in range(len(Model.interfaceModels[i].variableNames)):
                    self.myvariableNames.append("{0}_{1}_{2}_{3}".format(Model.interfaceModels[i].variableNames[k],i,j,self.id)) #Adds all the index necessary to make sure the name is unique in the list and avoid conflict
                    self.myunits.append(Model.interfaceModels[i].variableUnits[k])
                    if self.name != '':
                        self.mydescriptions.append("{0} : {1}{2}\n".format(self.name,Model.interfaceModels[i].variableDescriptions[k],j))
                    else:
                        self.mydescriptions.append("{0}{1}\n".format(Model.interfaceModels[i].variableDescriptions[k],j))
        try:
            self.variableDictionary = dict(zip(self.myvariableNames,self.param))
        except:
            print(traceback.format_exc())
            pass

    def change_param(self,param, variables, down=None,up=None):
        if param.ndim == 0:
            self.param = np.array([param])
        else:
            self.param = param
        if down is not None:
            self.down = down
        if up is not None:
            self.up = up
        if self.param is not None:          
            try:
                self.variableDictionary = dict(zip(variables,self.param))
            except:
                print(traceback.format_exc())
                pass
    def variableNames(self):
        return self.myvariableNames
    def variableUnits(self):
        return self.myunits
    def variableDescriptions(self):
        return self.mydescriptions


# =============================================================================

class Layers:
    def __init__(self,layers = [], interfaces = []):
        self.layers = layers # list of object of type Layer. size should be nlayers
        self.nlayers = len(layers)
        if interfaces == []:
            self.interfaces = [Interface() for i in range(self.nlayers+1)]
        else:
            self.interfaces = interfaces # list of object of type Interface. size should be nlayers+1
        self.names = []
        for layer in self.layers:
            self.names.append(layer.material.name)

    def set_FP(self, id_fp):
        self.index_FP = 1-id_fp # 1 (True) if Yes, 0 (False) is No

    def update_ini(self,other_layers):
        refresh = True
        if self.names == other_layers.names:
            refresh = False
            other_layerlist = other_layers.layers
            for i in range(self.nlayers):
                self.layers[i].update_ini(other_layerlist[i])
        else:
            self.layers = other_layers.layers
            self.interfaces = other_layers.interfaces
            self.nlayers = len(self.layers)
            self.names = []
            for layer in self.layers:
                self.names.append(layer.material.name)
        return refresh
        
    
    # transmission and reflexions coefficients depends on interfaces and adjacent layers
    def coefficients(self,w):
        self.refractive_indexes_with_air = np.ones([self.nlayers+2,len(w)],dtype = np.complex128)
        self.refractive_indexes_with_air[1:self.nlayers+1] = [np.sqrt(self.layers[i].material.epsilon(w)) for i in range(self.nlayers)]
        
        self.rf = np.zeros([self.nlayers+1,len(w)],dtype = np.complex128) # forward reflexion
        self.rb = np.zeros([self.nlayers+1,len(w)],dtype = np.complex128) # backward reflexion
        self.tf = np.ones([self.nlayers+1,len(w)],dtype = np.complex128) # forward transmission
        self.tb = np.ones([self.nlayers+1,len(w)],dtype = np.complex128) # forward transmission
        n = self.refractive_indexes_with_air # to make the calculs easier to read over
        
        for i in range(self.nlayers+1):
            self.rf[i] = (n[i+1]-n[i])/(n[i+1]+n[i])
            self.rb[i] = -self.rf[i]
            self.tf[i] = (2*n[i])/(n[i+1]+n[i])
            self.tb[i] = (2*n[i+1])/(n[i+1]+n[i])
            if self.interfaces[i].isMetasurface==1:
                H=0
                for k in range(len(Model.interfaceModels)):
                    for j in range(self.interfaces[i].nbTerms[k]):
                        paramList=[self.interfaces[i].variableDictionary.get("{0}_{1}_{2}_{3}".format(variableName,k,j,self.interfaces[i].id)) for variableName in Model.interfaceModels[k].variableNames]
                        H+=Model.interfaceModels[k].H(w,paramList)
                tf_t= self.tf[i]*(1-H)
                rf_t= self.rf[i] - self.tb[i]*H
                tb_t= self.tb[i]*(1-H)
                rb_t= self.rb[i] + self.tf[i]*H
                self.tf[i]=tf_t
                self.tb[i]=tb_t
                self.rf[i]=rf_t
                self.rb[i]=rb_t
            
    def transferfunction(self,w,delay_guess = 0,leftover_guess = np.zeros(2)): #may have to be seriously mmodified for metasurfaces, as I've worked under the assumption that rij = -rji ans things like that <- done for 1 layer, to do for the othe cases
        self.coefficients(w)
        if self.nlayers == 1:
            layer = self.layers[0]
            thickn = layer.thickness
            material = layer.material
            ref_index = np.sqrt(material.epsilon(w))
            
#            t12=2/(1+ref_index)
#            t21=2*ref_index/(1+ref_index)
#            r22=(ref_index-1)/(1+ref_index)
#            r22b=r22
#            rr=r22*r22b
#            tt=t12*t21

#            self.coefficients(w)
            ref_index = self.refractive_indexes_with_air[1]
            rr = self.rb[0]*self.rf[1]      # ras rsa
            tt = self.tf[0]*self.tf[1]      # tair->sample*tsample->air
            
            propa=np.exp(-j*w*(ref_index)*thickn/c)
            propaair=np.exp(-j*w*thickn/c)

            #Récupérer le contenu de File_FP
            f=open(os.path.join("temp",'temp_file_FP.bin'),'rb')
            id_FP = pickle.load(f)
            f.close()
            self.index_FP = 1 - id_FP # 1 (True) if Yes, 0 (False) is No

            if self.index_FP == 1 :
                FP=1/(1-rr*(propa**2))
            else : 
                FP=1
            

            coef = np.zeros(2)
            for i in range(0,len(leftover_guess)):
                coef[i] = leftover_guess[i]
            wnorm = w*1e-12
            leftnoise = (np.ones(len(wnorm))-(coef[0]*np.ones(len(wnorm))+coef[1]*(j*wnorm)**2)) 
            Z = tt*propa*(FP/propaair)*np.exp(j*w*delay_guess)*leftnoise #delay
            #print('Z')
            #print(Z)
            return Z
            
        # one should implement analytical solution for 2 layers for greater performance
        
        if self.nlayers == 3:         
            d1 = self.layers[0].thickness
            dS = self.layers[1].thickness
            d2 = self.layers[2].thickness
            
            delta1 = j*w*self.refractive_indexes_with_air[1]*d1/c #[0] is air, like [4]
            deltaS = j*w*self.refractive_indexes_with_air[2]*dS/c
            delta2 = j*w*self.refractive_indexes_with_air[3]*d2/c
            
            tm1 = 1
            tm2 = np.exp(-2*delta1)                 *self.rf[0]*self.rf[1] #rA1 r1S
            tm3 = np.exp(-2*delta2)                 *self.rf[2]*self.rf[3] #rS2 r2A 
            tm4 = np.exp(-2*delta1-2*delta2)        *self.rf[0]*self.rf[1]*self.rf[2]*self.rf[3] #rA1 r1S rS2 r2A
            tm5 = np.exp(-2*deltaS-2*delta2)        *self.rf[1]*self.rf[3] #r1S r2A
            tm6 = np.exp(-2*delta1-2*deltaS-2*delta2)*self.rf[0]*self.rf[3] #rA1 r2A
            tm7 = np.exp(-2*deltaS)                 *self.rf[1]*self.rf[2] #r1S rS2
            tm8 = np.exp(-2*delta1-2*deltaS)        *self.rf[0]*self.rf[2] #rA1 rS2
            
            propa = self.tf[0]*self.tf[1]*self.tf[2]*self.tf[3]*np.exp(-delta1-deltaS-delta2+j*w*(d1+dS+d2)/c) #propagation in air has been taken into account
            
            Z = propa/(tm1+tm2+tm3+tm4+tm5+tm6+tm7+tm8)*np.exp(j*w*delay_guess)
# =============================================================================
 #            tm1 = np.exp(+delta1+deltaS+delta2)
 #            tm2 = np.exp(-delta1+deltaS+delta2)*self.rf[0]*self.rf[1] #rA1 r1S
 #            tm3 = np.exp(+delta1+deltaS-delta2)*self.rf[2]*self.rf[3] #rS2 r2A
 #            tm4 = np.exp(-delta1+deltaS-delta2)*self.rf[0]*self.rf[1]*self.rf[2]*self.rf[3] #rA1 r1S rS2 r2A
 #            tm5 = np.exp(+delta1-deltaS-delta2)*self.rf[1]*self.rf[3] #r1S r2A
 #            tm6 = np.exp(-delta1-deltaS-delta2)*self.rf[0]*self.rf[3] #rA1 r2A
 #            tm7 = np.exp(+delta1-deltaS+delta2)*self.rf[1]*self.rf[2] #r1S rS2
 #            tm8 = np.exp(-delta1-deltaS+delta2)*self.rf[0]*self.rf[2] #rA1 rS2           
 #            Z = tt/(tm1+tm2+tm3+tm4+tm5+tm6+tm7+tm8)
# =============================================================================
            return Z

        else: # General case, consider changing it to S-parameters for better convergence
            tt = 1
            d = 0
            prop = 0
            T = []
            P = []
            l = len(w)
            for i in range(self.nlayers):
                 prop = prop-j*w*self.refractive_indexes_with_air[i+1]*self.layers[i].thickness/c
            for i in range(self.nlayers+1):
                tt = tt*self.tf[i]
                T.append(np.array([[np.ones(l),self.rf[i]],
                                    [self.rf[i],np.ones(l)]]))
            for i in range(self.nlayers):
                d = d+self.layers[i].thickness
                P.append(np.array([[np.ones(l), np.zeros(l)],
                                    [np.zeros(l),np.exp(-2*j*w*self.refractive_indexes_with_air[i+1]*self.layers[i].thickness/c)]]))
            M = T[0]
            for i in range(self.nlayers):
                M = multiply(M,P[i],l)
                M = multiply(M,T[i+1],l)
            Z = tt*np.exp(prop+j*w*d/c)/M[0][0] * np.exp(j*w*delay_guess)
            return Z
    
def multiply(A,B,l):
    " Takes two 2*2*l matrix and returns 2*2*l matrix"
    M = np.zeros([2,2,l],dtype = np.complex128)
    M[0][0] = A[0][0]*B[0][0]+A[0][1]*B[1][0]
    M[0][1] = A[0][0]*B[0][1]+A[0][1]*B[1][1]
    M[1][0] = A[1][0]*B[0][0]+A[1][1]*B[1][0]
    M[1][1] = A[1][0]*B[0][1]+A[1][1]*B[1][1]
    return M

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
