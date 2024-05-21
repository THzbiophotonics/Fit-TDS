#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math
import pickle
import subprocess
from pyswarm import pso   ## Library for optimization
import random
import numpy as np   ## Library to simplify the linear algebra calculations
import scipy.optimize as optimize  ## Library for optimization
import matplotlib.pyplot as plt ## Library for plotting results
from scipy.optimize import curve_fit ## Library for optimization
from epsillon3 import dielcal ## Library for resolving the inverse problem in our case (see the assumptions necessary to use this library)
from epsillonlayers8 import dielcal2
import h5py
from collections import Counter

import fit_TDSf as TDS
import fit_TDSm as Model

import warnings
#warnings.filterwarnings("ignore") #this is just to remove the 'devided by zero' runtime worning for low frequency
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

class ControlerBase:
    def __init__(self):
        self.clients_tab0 = list()
        self.clients_tab1 = list()
        self.clients_tab2 = list()
        self.clients_tab3 = list()
        self.message = ""

    def addClient0(self, client):
        self.clients_tab0.append(client)
    
    def addClient(self, client):
        self.clients_tab1.append(client)

    def addClient2(self, client):
        self.clients_tab2.append(client)

    def addClient3(self, client):
        self.clients_tab3.append(client)
        
    def refreshAll0(self, message):
        self.message = message
        for client in self.clients_tab0:
            client.refresh()

    def refreshAll(self, message):
        self.message = message
        for client in self.clients_tab1:
            client.refresh()

    def refreshAll2(self, message):
        self.message = message
        for client in self.clients_tab2:
            client.refresh()

    def refreshAll3(self, message):
        self.message = message
        for client in self.clients_tab3:
            client.refresh()


class Controler(ControlerBase):
    def __init__(self):
        super().__init__()
        # Creation:
        self.nbTerms0 = []
        self.material0=None
        self.nb_param0 = None
        self.myvariables0 = None
        self.mydescription0 = None
        self.myunits0 = None
        # Initialisation:
        self.nlayers = None
        self.nfixed_material = None
        self.noptim_material = None
        self.nfixed_metasurface = None
        self.noptim_metasurface = None
        self.materialList = []
        self.materialNames = []
        self.materials_param=[] 
        self.mylayers=None
        self.layerlist=None
        self.distinctInterfaceList = []
        self.interfaceList = []

        #To take into consideration Fabry-Perot effect, or not, in dielcal functions
        self.FP = []
        self.nbpi= None
        
        self.epsilonTarget = None
        
        self.datawithsample=None    ## We load the signal of the measured pulse with sample
        self.myreferencedata=TDS.inputdatafromfile
        self.myreferencedata.timeAndPulse=None ## We load the data of the measured reference pulse
        self.myreferencedata.Pulseinit=None
        self.myreferencedata.Spulseinit=None  ## We compute the spectrum of the measured reference pulse
        
        self.myglobalparameters=TDS.globalparameters
        self.myglobalparameters.t=None #this assumes input files are in ps ## We load the list with the time of the experiment
        self.myglobalparameters.freq = None        ## We create a list with the frequencies for the spectrum
        self.myglobalparameters.w=None
        
        self.nsample=None
        self.dt=None   ## Sample rate
        self.nb_param=None
        self.myvariables=None
        self.myunits=None
        self.mydescription=None
        self.mesparam=None
        self.myfitteddata=None
        self.previewdata=None
        self.fictionaldata=None
        self.errorIndex=0
        self.normalisedWeight=None
        self.normalisedNoise=None
        self.noisematrix = None
        
        ## parameters for the optimization algorithm
        self.swarmsize=1000
        self.maxiter=20

        # Variables for existence of temp Files
        self.is_temp_file_0 = 0 # temporary file storing material parameters
        self.is_temp_file_1 = 0 # temp file storing parameter choices
        self.is_temp_file_2 = 0
        self.is_temp_file_3 = 0 # temp file storing optimization results
        self.is_temp_file_4 = 0
        self.is_temp_file_5 = 0 # temp file storing algorithm choices

        # Variable to see if initialisation is done
        self.initialised = 0
        
# =============================================================================
# Creation tab
# =============================================================================
    def invalid_n_model0(self,msg):
        self.refreshAll0(msg + " \n")
    def error_message_path0(self):
        self.refreshAll0("Error: Please enter a valid path.")
    def ploting_text0(self,message):
        self.refreshAll0(message)
    def no_temp_file_0(self):
        self.refreshAll0("Unable to execute without running model choices first.")
        
    def material_parameters(self,nbTerms, isInterface):
        """Creates variable list and save choices in temp file when a model of material is submitted"""
        self.nbTerms0 = nbTerms
        if not isInterface:
            self.material0 = TDS.Material(nbTerms = nbTerms)
        else:
            self.material0 = TDS.Interface(isMetasurface = 1, nbTerms = nbTerms)
        self.myvariables0 = self.material0.variableNames()
        self.myunits0 = self.material0.variableUnits()
        self.mydescription0 = self.material0.variableDescriptions()
        
        self.nb_param0 = len(self.myvariables0)

        # Save in a temporal file the model choices for the optimizatoin
        mode_choicies_opt = self.nbTerms0

        if not os.path.isdir("temp"):
            os.mkdir("temp")

        f=open(os.path.join("temp",'temp_file_0.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_0 = 1
        self.refreshAll0("")
    
        
    def save_material_param(self,material,directory):
        """save parameters values and model choices in a file to be used as fixed material/metasurface"""
        headerChoicies = material.header
        np.savetxt(os.path.join(directory,material.name),material.param,header = headerChoicies)
    
    
# =============================================================================
# Initialisation tab
# =============================================================================

    def init(self):
        self.refreshAll("Initialisation: Ok")
        
    def error_message_path(self):
        self.refreshAll("Error: Please enter a valid path.")

    def loading_text(self):
        self.refreshAll("\n Processing... \n")

    def choices_ini(self,path_without_sample,path_with_sample,nlayers,nfixed_material,
                    noptim_material,nfixed_metasurface, noptim_metasurface,Lfiltering_index, 
                    Hfiltering_index, zeros_index, dark_index, cutstart, cutend,sharpcut, slope, intercept, fitDelay, delaymax_guess, delay_limit, delayfixed, modesuper, fitLeftover, leftcoef_guess, leftcoef_limit, leftfixed):
        """Process all the informations given in the first panel of initialisation: 
            create instances of classes to store data, apply filters, store choices in temp file 1_ini.
            Note that data is reload in creation of class myfitdata and before optimization, 
            any operations on data, like filters, has to be applied in the 3 cases."""
        self.pathwithoutsample = path_without_sample
        self.pathwithsample = path_with_sample
        self.fitDelay = fitDelay
        self.delaymax_guess = delaymax_guess
        self.delay_limit = delay_limit
        self.fitLeftover = fitLeftover
        self.leftcoef_guess = leftcoef_guess
        self.leftcoef_limit = leftcoef_limit
        self.leftfixed = leftfixed
        self.delayfixed = delayfixed

        self.datawithsample = np.loadtxt(self.pathwithsample)    ## We load the signal of the measured pulse with sample
        self.myreferencedata = TDS.inputdatafromfile(self.pathwithoutsample)
        
        self.myglobalparameters.t = self.myreferencedata.timeAndPulse[:,0]*1e-12 # this assumes input files are in ps ## We load the list with the time of the experiment
        self.nsample = len(self.myglobalparameters.t)
        self.dt=self.myglobalparameters.t.item(2)-self.myglobalparameters.t.item(1)  ## Sample rate
        self.myglobalparameters.freq = np.fft.rfftfreq(self.nsample, self.dt)        ## We create a list with the frequencies for the spectrum
        self.myglobalparameters.w = self.myglobalparameters.freq*2*np.pi
        
        self.myinputdata = TDS.mydata(self.datawithsample[:,1],self.myreferencedata.Spulseinit)    ## We create a variable containing the data related to the measured pulse with sample
        
        if modesuper == 1:
            self.mode = "superresolution"
            frep=99.991499600e6 # repetition frequency of the pulse laser used in the tds measurments in Hz, 99
            nsampleZP=np.round(1/(frep*self.dt)) #number of time sample betwen two pulses. IT has to be noted that it could be better to have an integer number there then the rounding does not change much
            self.nsamplenotreal=nsampleZP.astype(int)
            self.myglobalparameters.t=np.arange(nsampleZP)*self.dt  # 0001 #
            self.myglobalparameters.freq = np.fft.rfftfreq(self.nsamplenotreal, self.dt)
            self.myglobalparameters.w = 2*np.pi*self.myglobalparameters.freq
            
            self.myreferencedata.Pulseinit=np.pad(self.myreferencedata.timeAndPulse[:,1],(0,self.nsamplenotreal-self.nsample),'constant',constant_values=(0))
            self.myreferencedata.Spulseinit=(TDS.fft_gpu((self.myreferencedata.Pulseinit)))    # fft computed with GPU

            self.myinputdata=TDS.mydata(np.pad(self.datawithsample[:,1],(0,self.nsamplenotreal-self.nsample),'constant',constant_values=(0)),self.myreferencedata.Spulseinit)
        else:
            self.mode = "basic"
            self.nsamplenotreal = self.nsample
        
        # if one changes defaults values in TDSg this also has to change:
        self.Lfiltering = Lfiltering_index # 1 - Lfiltering_index 
        self.Hfiltering = Hfiltering_index # 1 - Hfiltering_index
        self.set_to_zeros = zeros_index # 1 - zeros_index
        self.dark_ramp = dark_index
        
        # Filter data
        Freqwindowstart = np.ones(len(self.myglobalparameters.freq))
        Freqwindowend = np.ones(len(self.myglobalparameters.freq))
        if self.Lfiltering:
            stepsmooth = cutstart/sharpcut
            Freqwindowstart = 0.5+0.5*np.tanh((self.myglobalparameters.freq-cutstart)/stepsmooth)
        if self.Hfiltering:
            #cutend = comm.bcast(cutend,root=0)
            #sharpcut = comm.bcast(sharpcut,root=0)
            stepsmooth = cutend/sharpcut
            Freqwindowend = 0.5-0.5*np.tanh((self.myglobalparameters.freq-cutend)/stepsmooth)
        self.Freqwindow = Freqwindowstart*Freqwindowend
        self.myreferencedata.Spulseinit = self.myreferencedata.Spulseinit*self.Freqwindow
        self.myinputdata.Spulse         = self.myinputdata.Spulse        *self.Freqwindow
        self.myreferencedata.Pulseinit  = np.fft.irfft(self.myreferencedata.Spulseinit, n = len(self.myreferencedata.Pulseinit))
        self.myinputdata.pulse          = np.fft.irfft(self.myinputdata.Spulse, n = len(self.myinputdata.pulse ) )


        self.timeWindow = np.ones(self.nsamplenotreal)
        if self.dark_ramp:
            #Enlève la rampe du dark noise du signal
            #for k in range(self.nsample):
            self.myreferencedata.Pulseinit = self.myreferencedata.Pulseinit - slope*self.myglobalparameters.t*1e12+intercept
            self.myinputdata.pulse = self.myinputdata.pulse - slope*self.myglobalparameters.t*1e12+intercept
            
        if self.set_to_zeros:
            #Remplace la fin du pulse de reference par des 0 (de la longueur du decalage entre les 2 pulses)
            imax1 = np.argmax(self.myreferencedata.Pulseinit)
            imax2 = np.argmax(self.myinputdata.pulse)
            tmax1 = self.myglobalparameters.t[imax1]
            tmax2 = self.myglobalparameters.t[imax2]
            deltaTmax = tmax2-tmax1
            
            tlim1 = self.myglobalparameters.t[self.nsample-1]-(5*deltaTmax/4)
            tlim2 = self.myglobalparameters.t[self.nsample-1]-(deltaTmax)
            for k in range(self.nsample):
                if self.myglobalparameters.t[k] < tlim1:
                     self.timeWindow[k] = 1
                elif self.myglobalparameters.t[k] >= tlim2:
                     self.timeWindow[k] = 0
                else:
                     term = (4/deltaTmax)*(self.myglobalparameters.t[k]-tlim1)
                     self.timeWindow[k] = 1-(3*term**2-2*term**3)
            self.myreferencedata.Pulseinit = self.myreferencedata.Pulseinit*self.timeWindow
            self.myreferencedata.Spulseinit = (np.fft.rfft((self.myreferencedata.Pulseinit)))
        
        self.nlayers = nlayers
        self.nfixed_material = nfixed_material
        self.noptim_material = noptim_material
        self.nfixed_metasurface = nfixed_metasurface
        self.noptim_metasurface = noptim_metasurface
        
        # files for choices made
        mode_choicies_opt=[self.pathwithoutsample, self.pathwithsample, 
                           self.Freqwindow,self.timeWindow,self.fitDelay, self.delaymax_guess, self.delay_limit, self.delayfixed, self.mode, self.fitLeftover, self.leftcoef_guess, self.leftcoef_limit, self.leftfixed]
#        [self.myinputdata, self.myreferencedata, self.myglobalparameters, self.nsample, self.delaymax, self.mode]

        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_1_ini.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()


    def param_ini(self,layerlist,position_optim_material,position_optim_thickness,position_optim_interface,id_FP, nbpi):
        """Process information given in last panel of Initialisation tab, updates temp file 2 if it already exists"""
        self.position_optim_material = position_optim_material
        self.position_optim_thickness = position_optim_thickness
        self.position_optim_interface = position_optim_interface
        self.FP= id_FP
        #Transmit the FP index in a file
        f=open(os.path.join("temp",'temp_file_FP.bin'),'wb')
        pickle.dump(id_FP,f,pickle.HIGHEST_PROTOCOL)
        f.close()

        
        if self.layerlist == None:
            self.layerlist = layerlist
            self.mylayers = TDS.Layers(layerlist,self.interfaceList)
            self.mylayers.set_FP(id_FP)
            
        else:
            new_layers = TDS.Layers(layerlist,self.interfaceList)
            new_layers.set_FP(id_FP)
            refresh = self.mylayers.update_ini(new_layers)
            self.layerlist = self.mylayers.layers
            self.nlayers = len(self.layerlist)
            if self.is_temp_file_2:
                f=open(os.path.join("temp",'temp_file_2.bin'),'wb')
                pickle.dump([self.position_optim_thickness,self.position_optim_material,
                             self.position_optim_interface,self.mylayers,self.layerlist,
                             self.interfaceList, self.mesparam],f,pickle.HIGHEST_PROTOCOL)
                f.close()
            if refresh:
                self.is_temp_file_2 = 0
                if self.nb_param:
                    self.nb_param = 0
        self.refreshAll2('')
        
        global c,j
        if (self.nlayers == 1):
            layer = self.layerlist[0]
            thickness=layer.thickness

            # calculating the delay to infer the index
            #self.epsilonTarget=dielcal(self.myinputdata.mytransferfunction,thickness,self.myglobalparameters) ## We search for the dielectric function using what we measured
            
            self.epsilonTarget=dielcal(self.myinputdata.mytransferfunction,thickness,self.myglobalparameters,self.FP, nbpi) ## We search for the dielectric function using what we measured
            self.deltaT=self.myglobalparameters.t[np.argmax(self.myinputdata.pulse)]-self.myglobalparameters.t[np.argmax(self.myreferencedata.Pulseinit)] #retard entre les deux max
            self.deltaTTT=self.myglobalparameters.t[np.argmin(self.myinputdata.pulse)]-self.myglobalparameters.t[np.argmin(self.myreferencedata.Pulseinit)] ## retard entre les deux min
            self.deltaTT=(np.sum(np.square(self.myinputdata.pulse)*self.myglobalparameters.t)/np.sum(np.square(self.myinputdata.pulse))-
                          np.sum(np.square(self.myreferencedata.Pulseinit)*self.myglobalparameters.t)/np.sum(np.square(self.myreferencedata.Pulseinit)))   #retard entre les deux barycentre, attention pour que ca fonctionne il faut que le rapport signal bruit soit le meme dans les deux cas !!
    
            self.refreshAll("Delay between the two maxima of the pulses:")
            self.refreshAll('delta T = {0}'.format(self.deltaT))
            self.refreshAll('n = {0}'.format(1+self.deltaT*c/thickness)) #indice qui en derive
            self.refreshAll('epsillon =  {0} \n'.format(np.square(1+self.deltaT*c/thickness))) #indice qui en derive
    
            self.refreshAll("Delay between the two minima of the pulses:")
            self.refreshAll('delta T = {0}'.format(self.deltaTTT))
            self.refreshAll('n = {0}'.format(1+self.deltaTTT*c/thickness)) #indice qui en derive
            self.refreshAll('epsillon = {0} \n'.format(np.square(1+self.deltaTTT*c/thickness))) #indice qui en derive
    
            self.refreshAll("Delay between the two energy barycenter of the pulses\n (beware that noise brings it to the middle for each one):")
            self.refreshAll('delta T= {0}'.format(self.deltaTT))
            self.refreshAll('n = {0}'.format(self.deltaTT*c/thickness)) #indice qui en derive
            self.refreshAll('epsillon = {0} \n'.format(np.square(self.deltaTT*c/thickness))) #indice qui en derive

# =============================================================================
# Model parameters
# =============================================================================
    def reset_values(self):
        self.myvariables=[]
        self.myunits=[]
        self.mydescription=[]
    
                                    
    def parameters_values(self,nbTerms):
        """Creates variable list and save choices in temp file when model of material to optimize is submitted"""
        self.nbTerms=nbTerms
        
        if self.myvariables is None:
            self.myvariables=[]
            self.myunits=[]
            self.mydescription=[]

        j=0
        for material in self.materialList:
            if material.fit_material == 1:
                material.change_variables(nbTerms[j])
                self.myvariables = self.myvariables+material.variableNames()
                self.myunits = self.myunits + material.variableUnits()
                self.mydescription = self.mydescription + material.variableDescriptions()
                j+=1
        for interface in self.distinctInterfaceList:
            if interface.fit_metasurface == 1:
                interface.change_variables(nbTerms[j])
                self.myvariables = self.myvariables+interface.variableNames()
                self.myunits = self.myunits + interface.variableUnits()
                self.mydescription = self.mydescription + interface.variableDescriptions()
                j+=1
        self.mylayers = TDS.Layers(self.layerlist,self.interfaceList)
        self.nb_param = len(self.myvariables)
        

        # Save in a temporal file the model choices for the optimization
        mode_choicies_opt=[self.myvariables, self.epsilonTarget]

        if not os.path.isdir("temp"):
            os.mkdir("temp")

        f=open(os.path.join("temp",'temp_file_1.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_1 = 1
        self.refreshAll2("")
    
    def invalid_n_model(self, msg):
        self.refreshAll2(msg + " \n")
   
    def invalid_swarmsize(self):
        self.refreshAll3("Invalid swarmsize. \n")

    def invalid_niter(self):
        self.refreshAll3("Invalid number of iterations. \n")

    def invalid_param(self):
        self.refreshAll2("Invalid parameters, try running the initialisation again. \n")

    def invalid_tun_opti_first(self):
        self.refreshAll2("Run the initialisation first. \n")
        
    def error_message_path2(self):
        self.refreshAll2("Error: Please enter a valid path.")


    def save_optimisation_param(self,mesparam):
        """Stores values submitted"""
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        self.mesparam = mesparam
        self.mylayers = TDS.Layers(self.layerlist,self.interfaceList)
        f=open(os.path.join("temp",'temp_file_2.bin'),'wb')
        pickle.dump([self.position_optim_thickness,self.position_optim_material,self.position_optim_interface, self.mylayers,self.layerlist,self.interfaceList, self.mesparam],f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_2 = 1
    
    def save_optimisation_param_outside(self,mesparam,directory,name):
        """Save values submitted in external file to reuse them later"""
        np.savetxt(os.path.join(directory,name),mesparam)
        
# =============================================================================
# Optimization
# =============================================================================
        
    def algo_parameters(self,choix_algo,swarmsize,niter,errorIndex,errorFile):
        """Save algorithm choices in temp file 5"""
        self.algo=choix_algo
        if errorIndex == 0:
            errorFile = None
        else:
            errorweightdata = h5py.File(errorFile, 'r')
            nameerr = list(errorweightdata.keys())[0]
            self.noisematrix = list(errorweightdata[nameerr])
            
        mode_choicies_opt=[choix_algo,int(swarmsize),int(niter),errorIndex,errorFile]
        if not os.path.isdir("temp"):
            os.mkdir("temp")

        f=open(os.path.join("temp",'temp_file_5.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_5 = 1

        self.refreshAll3("")
        
        
    def begin_optimization(self,nb_proc):
        """Run optimization and update layers"""
        output=""
        error=""
        returncode=0
        if sys.platform=="win32" or sys.platform=="cygwin":
            print("OS:Windows \n")
            if not os.path.isdir("temp"):
                os.mkdir("temp")
            optimization_filename = os.path.join('temp',"optimization.bat")
            try:
                with open(optimization_filename, 'w') as OPATH:
                   OPATH.writelines(['call set Path=%Path%;C:\ProgramData\Anaconda3 \n',
                   'call set Path=%Path%;C:\ProgramData\Anaconda3\condabin \n',
                   'call set Path=%Path%;C:\ProgramData\Anaconda3\Scripts \n',
                   #'call conda activate \n',
                   'call mpiexec -n {0} python optimization.py'.format(nb_proc)])
#                    OPATH.writelines([f'call mpiexec -n {nb_proc} optimization.exe'])
                subprocess.call(optimization_filename)
                returncode = 0
                error = ""
                output = ""
            except:
                print("No parallelization! You don't have MPI installed or there's a problem with your MPI.")
                with open(optimization_filename, 'w') as OPATH:
                    OPATH.writelines([f'call optimization.exe'])
                subprocess.call(optimization_filename)
                returncode = 0
                error = ""
                output = ""
        elif sys.platform=="linux" or sys.platform=="darwin":
            print("OS:Linux/MacOS \n")
            optimization_filename = os.path.join('temp',"optimization.sh")
            try:
                # Check if Open MPI is correctly installed
                try:
                    command = 'mpiexec --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_mpi,error_mpi = process.communicate()
                    returncode_mpi=process.returncode
                except:
                    returncode_mpi = 1
                    error_mpi = "Command mpiexec not recognized."

                try:
                    command = './py3-env/bin/python --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_py3,error_py3 = process.communicate()
                    returncode_py3=process.returncode
                    python_path = "./py3-env/bin/python"
                except:
                    try:
                        command = "python3 --version"
                        process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        output_py3,error_py3 = process.communicate()
                        returncode_py3=process.returncode
                        python_path = "python3"
                    except:
                        returncode_py3 = 1
                        error_py3 = "Command python3 not recognized."

                # Run optimization
                if returncode_mpi==0:
                    if returncode_py3==0:
                        # command = 'mpiexec -n {0} {1} optimization.py'.format(nb_proc, python_path)
                        command = f'mpiexec.mpich -n {nb_proc} python optimization.py'
                    else:
                        print("Problem with python command : \n {} \n".format(error_py3))
                        return(0)
                else:
                    print("No parallelization! You don't have MPI installed or there's a problem with your MPI: \n {}".format(error_mpi))
                    if returncode_py3==0:
                        command = '{0} optimization.py'.format(python_path)
                    else:
                        print("Problem with python command : \n {} \n".format(error_py3))
                        return(0)

                try:
                    with open(optimization_filename, 'w') as OPATH:
                        OPATH.writelines(command)
                    returncode = subprocess.call('chmod +x ./{}'.format(optimization_filename),shell=True)
                    if returncode == 0:
                        print(f"subprocess ran")
                        returncode = subprocess.call(f'{optimization_filename}',shell=True)
                        # returncode = subprocess.call('./{}'.format(optimization_filename),shell=True)
                    if returncode == 1:
                        command = ""
                        command = 'import subprocess \ncommand = "{0}" \nprocess = subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE) \noutput,error = process.communicate() \nprint("Output : " + str(output) + "\\n Error: " + str(error) + "\\n")'.format(command)
                        with open("launch_optimization.py", 'w') as OPATH:
                            OPATH.writelines(command)
                        try:
                            import launch_optimization
                            try:
                                f=open(os.path.join("temp",'temp_file_3.bin'),'rb')
                                f.close()
                                returncode=0
                            except:
                                print("Unknown problem.")
                                sys.exit()
                        except:
                            print("Unknown problem.")
                            sys.exit()
                except:
                    returncode = 1
                    error = "Unknow problem."
                    output = ""
            except:
                print("Unknown problem.")
                sys.exit()

        else:
            print("System not supported.")
            return(0)

        if returncode==0:
            f=open(os.path.join("temp",'temp_file_3.bin'),'rb')
            var_inter=pickle.load(f)
            f.close()
            self.is_temp_file_3 = 1
            xopt=var_inter[0]
            message=var_inter[1]
            delay_guess = 0
            leftover_guess = np.zeros(2)
            if any(i!=0 for i in self.leftcoef_guess):
                count=-1
                for i in range (0,len(self.leftcoef_guess)):                                          
                    count=count+1
                    leftover_guess[i] = xopt[-len(self.leftcoef_guess)+count]
            if self.delaymax_guess !=0:
                if any(i!=0 for i in self.leftcoef_guess):
                    delay_guess = xopt[-len(self.leftcoef_guess)-1]
                else:
                    delay_guess = xopt[-1]
            for i in self.position_optim_material:
                self.layerlist[i].material.change_param(xopt,self.myvariables)
            for i in self.position_optim_interface:
                self.interfaceList[i].change_param(xopt,self.myvariables)
            for i, pos in enumerate(self.position_optim_thickness):
                self.layerlist[pos].thickness = xopt[self.nb_param+i]
            mylayers = TDS.Layers(self.layerlist,self.interfaceList)
            self.myfitteddata=TDS.myfitdata(mylayers,delay_guess=delay_guess,leftover_guess=leftover_guess)
            if self.errorIndex == 1:
                try:
                    weight = np.loadtxt(self.errorfile)
                    weightnorm = np.linalg.norm(weight)/np.linalg.norm(np.ones(self.nsamplenotreal))
                    self.normalisedWeight = weight/weightnorm
                except:
                    self.normalisedWeight = None
            else:
                self.normalisedWeight = None
            self.refreshAll3(message)
        else:
            self.refreshAll3("Output : \n {} \n".format(output))
            print("System not supported. \n")
            print('Output : \n {0} \n Error : \n {1} \n'.format(output, error))
            return(0)

    def loading_text3(self):
        self.refreshAll3("\n Processing... \n")

    def message_log_tab3(self,message):
        self.refreshAll3(message)

    def error_message_path3(self):
        self.refreshAll3("Error: Please enter a valid path.")

    def error_message_output_paths(self):
        self.refreshAll3("Invalid output paths.")

    def error_message_output_filename(self):
        self.refreshAll3("Invalid output filename.")

    def get_output_paths(self,outputdir,time_domain,frequency_domain,out_opt_filename):
        """Check output names and path and stores them in temp file 4 if they are valid. """
        try:
            self.outputdir = str(outputdir)
        except:
            self.refreshAll3("Invalid output directory.")
            return(0)
        try:
            self.time_domain = str(time_domain)
        except:
            self.refreshAll3("Invalid name for time domain output.")
            return(0)
        try:
            self.frequency_domain = str(frequency_domain)
        except:
            self.refreshAll3("Invalid name for frequency domain output.")
            return(0)
        try:
            self.out_opt_filename = str(out_opt_filename)
        except:
            self.refreshAll3("Invalid name for frequency domain output.")
            return(0)
        output_paths = [self.outputdir,self.time_domain,self.frequency_domain, self.out_opt_filename]
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_4.bin'),'wb')
        pickle.dump(output_paths,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_4 = 1

    def preview(self):
        """Creates data that will be plotted"""
        if self.is_temp_file_2 == 1:
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb') # enables preview to work between optmization and construction and pparameters change
            [position_optim_thickness, position_optim_material, position_optim_interface, mylayers,mylayerlist,myinterfacelist,mesparam]=pickle.load(f)
            f.close()
            self.previewdata=TDS.myfitdata(mylayers, self.delaymax_guess, self.leftcoef_guess)
            self.myfitteddata = None # Sinon on ne peut plus faire de preview apres avoir optimise, cf definition de refresh dans TDSg. Il y aurait d'autres manière de le faire.
            erreur = None
            pulsenorm = np.linalg.norm(self.myinputdata.pulse)
            input_reduced = self.myinputdata.pulse[:self.nsample] #input_reduced norm is equal to pulsenorm

            if self.errorIndex == 0:
                Z = mylayers.transferfunction(self.myglobalparameters.w, self.delaymax_guess, self.leftcoef_guess)
                if self.mode == "basic":
                    fit_pulse = np.fft.irfft(Z*self.myreferencedata.Spulseinit, n = len(self.myreferencedata.Pulseinit))
                    erreur=np.linalg.norm(fit_pulse-self.myinputdata.pulse)/pulsenorm
                else:
                    Spectrumtot=Z*self.myreferencedata.Spulseinit
                    pulse_theo=(np.fft.irfft((np.array(Spectrumtot)), n = len(self.myreferencedata.Pulseinit))) # calcul from calculedpulse. In fact it is the same calcul as in the basic mode for i!=0
                    pulse_theo_reduced = pulse_theo[:self.nsample]
                    erreur=np.linalg.norm(input_reduced-pulse_theo_reduced)/pulsenorm 
            if self.errorIndex == 1:
                try:
                    weight = np.loadtxt(self.errorFile)
                    try:
                        if len(weight[0]) == 2: #in case there is time
                            weight = weight[:,1]
                    except:
                        pass
                    weightnorm = np.linalg.norm(weight)/np.linalg.norm(np.ones(self.nsamplenotreal))
                    self.normalisedWeight = weight/weightnorm
                    self.normalisedNoise = None
                    self.noisematrix = None

                    Z = mylayers.transferfunction(self.myglobalparameters.w, self.delaymax_guess, self.leftcoef_guess)
                    if self.mode == "basic":
                        fit_pulse = np.fft.irfft(Z*self.myreferencedata.Spulseinit, n = len(self.myreferencedata.Pulseinit))
                        erreur=np.linalg.norm((fit_pulse-self.myinputdata.pulse)*self.normalisedWeight)/pulsenorm
                    else:
                        Spectrumtot=Z*self.myreferencedata.Spulseinit
                        pulse_theo=np.fft.irfft((np.array(Spectrumtot)),n = len(self.myreferencedata.Pulseinit))
                        pulse_theo_reduced = pulse_theo[:self.nsample]
                        erreur=np.linalg.norm((input_reduced-pulse_theo_reduced)*self.normalisedWeight)/pulsenorm
                except:
                    self.normalisedWeight = None
                    self.normalisedNoise = None
                    self.noisematrix = None
            elif self.errorIndex == 2:
                try:
                    noise = np.loadtxt(self.errorFile)
                    try:
                        if len(noise[0]) == 2: #in case there is time
                            noise = noise[:,1]
                    except:
                        pass
                    noisenorm = np.linalg.norm(noise)/np.linalg.norm(np.ones(self.nsamplenotreal))
                    self.normalisedNoise = noise/noisenorm
                    self.normalisedWeight = None
                    self.noisematrix = None
                    Z = mylayers.transferfunction(self.myglobalparameters.w, self.delaymax_guess, self.leftcoef_guess)
                    if self.mode == "basic":
                        fit_pulse = np.fft.irfft(Z*self.myreferencedata.Spulseinit, n = len(self.myreferencedata.Pulseinit))
                        erreur=np.linalg.norm((fit_pulse-self.myinputdata.pulse)/self.normalisedNoise)/pulsenorm
                    
                except:
                    self.normalisedWeight = None
                    self.normalisedNoise = None
                    self.noisematrix = None

            elif self.errorIndex == 3:
                try:
                    noisedata = h5py.File(self.errorFile, 'r')
                    name = list(noisedata.keys())[0]
                    self.noisematrix = list(noisedata[name])
                    if np.shape(self.noisematrix)[0] != self.nsample or np.shape(self.noisematrix)[1] != self.nsample:
                        self.refreshAll3('Please enter a valid path. The file should be a {} square matrix'.format(self.nsample))
                        return 0
                    self.normalisedWeight = None
                    self.normalisedNoise = None
                    #noisematnorm = np.linalg.norm(self.noisematrix)/np.linalg.norm(np.ones(self.nsamplenotreal)) #TODO
                    noisematnorm = np.linalg.norm(self.noisematrix)
                    normalisednoisemat = self.noisematrix/noisematnorm
                    Z = mylayers.transferfunction(self.myglobalparameters.w, self.delaymax_guess, self.leftcoef_guess)
                    if self.mode == "basic":
                        fit_pulse = np.fft.irfft(Z*self.myreferencedata.Spulseinit, n = len(self.myreferencedata.Pulseinit))
                        Rtls = np.dot(normalisednoisemat,(self.myinputdata.pulse-fit_pulse))
                        erreur = np.sqrt(np.dot(np.transpose(Rtls),Rtls))/pulsenorm

                except Exception as e:
                    print(e)
                    self.normalisedWeight = None
                    self.normalisedNoise = None
                    self.noisematrix = None
            else:
                self.normalisedWeight = None
                self.normalisedNoise = None
                self.noisematrix = None
            if erreur:
            	self.refreshAll3("The preview error is :" + str(erreur))
            	self.refreshAll3("Done")
            else:
            	self.refreshAll3("Weight or Noise is missing to print error")
        else:
            self.no_temp_file_2()
        
    def generateFictionalSample(self,tempstd=0,ampstd=0, name = 'fictionalsample',directory = None): #files
        """Creates fictionnal sample (preview + noise)"""
        if self.is_temp_file_2 == 1:
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb')
            [position_optim_thickness, position_optim_material, position_optim_interface, mylayers,mylayerlist,myinterfacelist,mesparam]=pickle.load(f)
            f.close()
            self.fictionaldata=TDS.myfitdata(self.mylayers)
            # creation of data with gaussian noise
            outputtime=np.column_stack(((self.myglobalparameters.t+[random.gauss(0,tempstd) for i in range(self.nsample)])*1e12,self.fictionaldata.pulse+[random.gauss(0,ampstd) for i in range(self.nsample)]))
            if directory!=None:
                np.savetxt(os.path.join(directory,'{}fiction'.format(name)),outputtime)
            else:
                np.savetxt('fictional{}'.format(name),outputtime)
            self.refreshAll3("Fictional sample saved")
        else:
            self.no_temp_file_2()

    def compute_eps_init(self, fileName = None):
        """Computes epsilon according to the thicknesses given in initialization."""
        if self.is_temp_file_2 == 1:
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb') # enables preview to work between optimization and construction and parameters change
            [position_optim_thickness, position_optim_material, position_optim_interface, mylayers,mylayerlist,myinterfacelist,mesparam]=pickle.load(f)
            f.close()
        else:
            self.no_temp_file_2()
            return(0)
        if self.nlayers == 1:
            if self.initialised:
                self.epsilonTarget=dielcal(self.myinputdata.mytransferfunction,
                                           mylayerlist[0].thickness,self.myglobalparameters,self.FP, nbpi)
                self.refreshAll3("Done")
            else:
                self.refreshAll3("PLease run initialization first")
        elif (self.nlayers == 3)&(self.nfixed_material == 1)&(fileName is not None):
            if self.is_temp_file_3 == 1:
                self.refreshAll3("Unable to run after optimization")
                return(0)
            if self.initialised:
                try:
                    emptydata = np.loadtxt(fileName)
                    myepsilondata = TDS.mydata(self.datawithsample[:,1],np.fft.rfft(emptydata[:,1]))
                    data=dielcal2(myepsilondata.mytransferfunction,
                                                mylayerlist[1].thickness,self.myglobalparameters,
                                                mylayerlist[0].material.epsilon(self.myglobalparameters.w),
                                                mylayerlist[0].thickness, mylayerlist[2].thickness, self.FP, nbpi)
                    self.epsilonTarget = data[0]
                    self.refreshAll3("Done")
                except:
                    self.refreshAll3("Please enter a matching file")
            else:
                self.refreshAll3("PLease run initialization first")
        else:
            self.refreshAll3("Unable to compute for other than one layer or full cuvette")
    
    def compute_eps_opti(self, fileName = None):
        """Computes epsilon according to the thicknesses found in optimization, if they were optimization parameters"""
        if self.nlayers == 1:
            if self.is_temp_file_3 == 1:
                f=open(os.path.join("temp",'temp_file_3.bin'),'rb')
                var_inter=pickle.load(f)
                f.close()
                xopt=var_inter[0]
                if self.layerlist[0].fit_thickness:
                    thickness = xopt[-1]
                    self.epsilonTarget=dielcal(self.myinputdata.mytransferfunction,
                                                 thickness,self.myglobalparameters,self.FP, nbpi)
                    self.refreshAll3("Done")
                else:
                    self.refreshAll3("Thickness was not used as a variable in the last optimization.")
                
            else:
                self.refreshAll3("Please run optimization first")
        elif (self.nlayers == 3)&(self.nfixed_material == 1)&(fileName is not None):
            if self.is_temp_file_3 == 1:
                try:
                    emptydata = np.loadtxt(fileName)
                    myepsilondata = TDS.mydata(self.datawithsample[:,1],np.fft.rfft(emptydata[:,1]))
                    data=dielcal2(myepsilondata.mytransferfunction,
                                                self.layerlist[1].thickness,self.myglobalparameters,
                                                self.layerlist[0].material.epsilon(self.myglobalparameters.w),
                                                self.layerlist[0].thickness, self.layerlist[2].thickness,self.FP, nbpi)
                    self.epsilonTarget = data[0]
                    self.refreshAll3("Done")
                except:
                    self.refreshAll3("Please enter a matching file")
        else:
            self.refreshAll3("Unable to compute for other than one layer")
    def compute_eps_phase_corraction(self, fileName=None):
        if self.is_temp_file_2 == 1:
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb') # enables preview to work between optimization and construction and parameters change
            [position_optim_thickness, position_optim_material, position_optim_interface, mylayers,mylayerlist,myinterfacelist,mesparam]=pickle.load(f)
            f.close()
        else:
            self.no_temp_file_2()
            return(0)
        if self.nlayers == 1:
            if self.initialised:
                f=open(os.path.join("temp",'temp_file_phase.bin'),'rb')
                nbpi= pickle.load(f)
                f.close()
                print("phase2:", nbpi)
                self.epsilonTarget=dielcal(self.myinputdata.mytransferfunction,
                                           mylayerlist[0].thickness,self.myglobalparameters,self.FP, nbpi)
                self.refreshAll3("Done")
            else:
                self.refreshAll3("PLease run initialization first")
        elif (self.nlayers == 3)&(self.nfixed_material == 1)&(fileName is not None):
            if self.is_temp_file_3 == 1:
                self.refreshAll3("Unable to run after optimization")
                return(0)
            if self.initialised:
                try:
                    emptydata = np.loadtxt(fileName)
                    myepsilondata = TDS.mydata(self.datawithsample[:,1],np.fft.rfft(emptydata[:,1]))
                    data=dielcal2(myepsilondata.mytransferfunction,
                                                mylayerlist[1].thickness,self.myglobalparameters,
                                                mylayerlist[0].material.epsilon(self.myglobalparameters.w),
                                                mylayerlist[0].thickness, mylayerlist[2].thickness, self.FP, nbpi)
                    self.epsilonTarget = data[0]
                    self.refreshAll3("Done")
                except:
                    self.refreshAll3("Please enter a matching file")
            else:
                self.refreshAll3("PLease run initialization first")
        else:
            self.refreshAll3("Unable to compute for other than one layer or full cuvette")

    def ploting_text3(self,message):
        self.refreshAll3(message)
        
    def no_temp_file_1(self):
        self.refreshAll3("Unable to execute without running step Initialization and model choices first.")

    def no_temp_file_2(self):
        self.refreshAll3("Unable to execute without running step 'model parameters' window first.")

    def no_temp_file_4(self):
        self.refreshAll3("Unable to execute without selecting path for output data first.")
    
    def no_temp_file_5(self):
        self.refreshAll3("Unable to execute without optimization parameters")
