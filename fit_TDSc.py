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
import fit_TDSf as TDS

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

from pyOpt import Optimization   ## Library for optimization
from pyOpt import ALPSO  ## Library for optimization
#from pyOpt import SLSQP  ## Library for optimization



class ControlerBase:
    def __init__(self):
        self.clients_tab1 = list()
        self.clients_tab2 = list()
        self.clients_tab3 = list()
        self.message = ""

    def addClient(self, client):
        self.clients_tab1.append(client)

    def addClient2(self, client):
        self.clients_tab2.append(client)

    def addClient3(self, client):
        self.clients_tab3.append(client)

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
        self.mesdata=None    ## We load the signal of the measured pulse with sample
        self.myinputdatafromfile=TDS.inputdatafromfile
        self.myglobalparameters=TDS.globalparameters
        self.myinputdatafromfile.PulseInittotal=None ## We load the data of the measured reference pulse
        self.myglobalparameters.t=None #this assumes input files are in ps ## We load the list with the time of the experiment
        self.nsample=None
        self.myinputdatafromfile.Pulseinit=None
        self.dt=None   ## Sample rate
        self.myglobalparameters.freq = None        ## We create a list with the frequencies for the spectrum
        self.myglobalparameters.w=None
        self.myinputdatafromfile.Spulseinit=None  ## We compute the spectrum of the measured reference pulse
        self.nb_param=None
        self.myvariables=None
        self.myunits=None
        self.mydescription=None
        self.mesparam=None
        self.myfitteddata=None
        self.previewdata=None

        ## parameters for the optimization algorithm
        self.swarmsize=1000
        self.maxiter=20

        # Variables for existence of temp Files
        self.is_temp_file_1 = 0
        self.is_temp_file_2 = 0
        self.is_temp_file_3 = 0
        self.is_temp_file_4 = 0

        # Variable to see if thickness of the sample was given by the user
        self.is_thickness = 0

    def error_message_init_values(self):
        self.refreshAll("Error: Please enter a real number.")

    def error_message_path(self):
        self.refreshAll("Error: Please enter a valid path.")

    def warning_negative_thickness(self):
        self.refreshAll("Warning: You entered a negative thickness")

    def warning_uncertainty(self):
        self.refreshAll("Warning: The uncertainty you entered it's not between 0 an 100%.")

    def loading_text(self):
        self.refreshAll("\n Processing... \n")

    def ploting_text(self,message):
        self.refreshAll(message)

    def ploting_text3(self,message):
        self.refreshAll3(message)


    def param_ini(self,thickness,uncertainty,path_without_sample,path_with_sample):
        global c,j
        self.z=thickness
        self.deltaz=uncertainty/100
        self.pathwithoutsample=path_without_sample
        self.pathwithsample=path_with_sample

        self.mesdata=np.loadtxt(self.pathwithsample)    ## We load the signal of the measured pulse with sample
        self.myinputdatafromfile.PulseInittotal=np.loadtxt(self.pathwithoutsample) ## We load the data of the measured reference pulse
        self.myglobalparameters.t=self.myinputdatafromfile.PulseInittotal[:,0]*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
        self.nsample=len(self.myglobalparameters.t)
        self.myinputdatafromfile.Pulseinit=self.myinputdatafromfile.PulseInittotal[:,1]
        self.dt=self.myglobalparameters.t.item(2)-self.myglobalparameters.t.item(1)   ## Sample rate
        self.myglobalparameters.freq = np.fft.rfftfreq(self.nsample, self.dt)        ## We create a list with the frequencies for the spectrum
        self.myglobalparameters.w=self.myglobalparameters.freq*2*np.pi
        self.myinputdatafromfile.Spulseinit=(np.fft.rfft((self.myinputdatafromfile.Pulseinit)))  ## We compute the spectrum of the measured reference pulse
        self.myinputdata=TDS.mydata(self.mesdata[:,1],self.myinputdatafromfile.Spulseinit,self.z,self.myglobalparameters)    ## We create a variable containing the data related to the measured pulse with sample
        self.monepsilon=dielcal(np.fft.rfft((self.mesdata[:,1]))/self.myinputdatafromfile.Spulseinit,self.z,self.myglobalparameters) ## We search for the dielectric function using what we measured
        # calculating the delay to infer the index
        self.deltaT=self.myglobalparameters.t[np.argmax(self.myinputdata.pulse)]-self.myglobalparameters.t[np.argmax(self.myinputdatafromfile.Pulseinit)] #retard entre les deux max
        self.deltaTTT=self.myglobalparameters.t[np.argmin(self.myinputdata.pulse)]-self.myglobalparameters.t[np.argmin(self.myinputdatafromfile.Pulseinit)] ## retard entre les deux min
        self.deltaTT=(np.sum(np.square(self.myinputdata.pulse)*self.myglobalparameters.t)/np.sum(np.square(self.myinputdata.pulse))-
                      np.sum(np.square(self.myinputdatafromfile.Pulseinit)*self.myglobalparameters.t)/np.sum(np.square(self.myinputdatafromfile.Pulseinit)))   #retard entre les deux barycentre, attention pour que ca fonctionne il faut que le rapport signal bruit soit le meme dans les deux cas !!

        self.refreshAll("Delay between the two maxima of the pulses:")
        self.refreshAll(f'delta T = {self.deltaT}')
        self.refreshAll(f'n = {1+self.deltaT*c/self.z}') #indice qui en derive
        self.refreshAll(f'epsillon =  {np.square(1+self.deltaT*c/self.z)} \n') #indice qui en derive

        self.refreshAll("Delay between the two minima of the pulses:")
        self.refreshAll(f'delta T = {self.deltaTTT}')
        self.refreshAll(f'n = {1+self.deltaTTT*c/self.z}') #indice qui en derive
        self.refreshAll(f'epsillon = {np.square(1+self.deltaTTT*c/self.z)} \n') #indice qui en derive

        self.refreshAll("Delay between the two energy barycenter of the pulses\n (beware that noise brings it to the middle for each one):")
        self.refreshAll(f'delta T= {self.deltaTT}')
        self.refreshAll(f'n = {self.deltaTT*c/self.z}') #indice qui en derive
        self.refreshAll(f'epsillon = {np.square(self.deltaTT*c/self.z)} \n') #indice qui en derive

    def parameters_values(self,choix_algo,mymodelstruct,thickness,isdrude,scattering,n,nDebye,swarmsize,niter):
        self.algo=choix_algo
        self.mymodelstruct=mymodelstruct
        self.zvariable=thickness
        self.isdrude=isdrude
        self.scattering=scattering
        self.n=int(n)
        self.nDebye=int(nDebye)
        if self.mymodelstruct==1:
            for i in range(0,1):
                self.myvariables=[f'Omega resonator/metasurface_{i}',
                                  f'Tau 0 resonator/metasurface_{i}',
                                  f'Tau 1 resonator/metasurface_{i}',
                                  f'Tau 2 resonator/metasurface_{i}',
                                  f'delta Theta resonator/metasurface_{i}']
                self.myunits=["Radian / s", "s" ,"s","s","Radian"]
                self.mydescription=[f'Central angular frequency of the mode of the resonator #{i}\n',
                                    f'Absorption life time of the mode of the resonator #{i}\n',
                                    f'Forward coupling lifetime of the mode of the resonator #{i}\n',
                                    f'Backward coupling lifetuime of the mode of the resonator #{i}\n',
                                    f'Phase between Forward and backward coupling for the resontator #{i}\n']
        else:
            self.myvariables=[]
            self.myunits=[]
            self.mydescription=[]


        if self.scattering == 0:
            self.myvariables=self.myvariables+["Beta"]
            self.myunits=self.myunits+["1/m"]
            self.mydescription=self.mydescription+["Loss coefficient"]
            self.myvariables=self.myvariables+["Scat_freq_min"]
            self.myunits=self.myunits+["Hz"]
            self.mydescription=self.mydescription+["Beginning frequency of scattering"]
            self.myvariables=self.myvariables+["Scat_freq_max"]
            self.myunits=self.myunits+["Hz"]
            self.mydescription=self.mydescription+["Ending frequency of scattering"]

        self.myvariables=self.myvariables+["epsillon_inf"]
        self.myunits=self.myunits+["dimensionless"]
        self.mydescription=self.mydescription+["Permitivity at very high frequency\n"]
        if self.isdrude==0:
            self.myvariables=self.myvariables+["Omega_p","gamma"]
            self.myunits=self.myunits+["radian/s","radian/s"]
            self.mydescription=self.mydescription+["Drude's Model Plasma frequency \n","Drude damping rate \n"]
        for i in range(0,self.n):
            self.myvariables=self.myvariables+[f'Delta_Epsillon_{i}', f'1/(2pi)*Omega0_{i}',f'1/(2pi)*Gamma_{i}']
            self.myunits=self.myunits+["dimensionless", "Hz","Hz"]
            self.mydescription=self.mydescription+["Oscillator strentgh of the mode # \n", f'Frequency of the mode #{i}\n',
                                                   f'Linewidth of the mode #{i}\n']

        for iDebye in range(0,self.nDebye):
            self.myvariables=self.myvariables+[f'Delta_Epsillon_{iDebye}', f'1/(2pi)*OmegaD_{iDebye}']
            self.myunits=self.myunits+["dimensionless", "Hz"]
            self.mydescription=self.mydescription+["Oscillator strentgh of the mode # \n", f'Frequency of the mode #{iDebye}\n']

        self.nb_param = len(self.myvariables)

        # Save in a temporal file the model choices for the optimizatoin
        mode_choicies_opt=[self.z, self.deltaz, self.pathwithoutsample,
                           self.pathwithsample,self.monepsilon, self.myvariables,
                           self.myunits, self.mydescription, choix_algo,
                           mymodelstruct, thickness, isdrude,scattering,int(n),int(nDebye),int(swarmsize),int(niter)]

        if not os.path.isdir("temp"):
            os.mkdir("temp")

        f=open(os.path.join("temp",'temp_file_1.bin'),'wb')
        pickle.dump(mode_choicies_opt,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_1 = 1

        self.refreshAll2("")

    def invalid_n_lorentz(self):
        self.refreshAll2("Invalid number of Lorentz Oscillators. \n")

    def invalid_n_debye(self):
        self.refreshAll2("Invalid number of Debye Oscillators. \n")

    def invalid_swarmsize(self):
        self.refreshAll2("Invalid swarmsize. \n")

    def invalid_niter(self):
        self.refreshAll2("Invalid number of iterations. \n")

    def invalid_param_opti(self):
        self.refreshAll2("Invalid parameters for optimization, try running the initialisation again. \n")

    def invalid_tun_opti_first(self):
        self.refreshAll2("Run the initialisation first. \n")


    def save_optimisation_param(self,mesparam):
        if not os.path.isdir("temp"):
            os.mkdir("temp")
        f=open(os.path.join("temp",'temp_file_2.bin'),'wb')
        pickle.dump(mesparam,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.is_temp_file_2 = 1


    def begin_optimization(self,nb_proc):
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
                    'call conda activate \n',
                    f'call mpiexec -n {nb_proc} python optimization.py'])
                subprocess.call(optimization_filename)
                returncode = 0
                error = ""
                output = ""
            except:
                print("No parallelization! You don't have MPI installed or there's a problem with your MPI.")
                with open(optimization_filename, 'w') as OPATH:
                    OPATH.writelines(['call set Path=%Path%;C:\ProgramData\Anaconda3 \n',
                    'call set Path=%Path%;C:\ProgramData\Anaconda3\condabin \n',
                    'call set Path=%Path%;C:\ProgramData\Anaconda3\Scripts \n',
                    'call conda activate \n',
                    f'call python optimization.py'])
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
                    command = 'python3 --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_py3,error_py3 = process.communicate()
                    returncode_py3=process.returncode
                except:
                    returncode_py3 = 1
                    error_py3 = "Command python3 not recognized."

                try:
                    command = 'python --version'
                    process=subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    output_py,error_py = process.communicate()
                    returncode_py=process.returncode
                except:
                    returncode_py = 1
                    error_py = "Command python not recognized."


                # Run optimization
                if returncode_mpi==0:
                    if returncode_py3==0:
                        command = f'mpiexec -n {nb_proc} python3 optimization.py'
                    elif returncode_py==0:
                        print('Python 3 not installed, trying with Python 2.')
                        command = f'mpiexec -n {nb_proc} python optimization.py'
                    else:
                        print("Problem with python3 command : \n {} \n".format(error_py3))
                        print("Problem with python command : \n {} \n".format(error_py))
                        return(0)
                else:
                    print("No parallelization! You don't have MPI installed or there's a problem with your MPI: \n {}".format(error_mpi))
                    if returncode_py3==0:
                        command = 'python3 optimization.py'
                    elif returncode_py==0:
                        print('Python 3 not installed, trying with Python 2.')
                        command = 'python optimization.py'
                    else:
                        print("Problem with python3 command : \n {} \n".format(error_py3))
                        print("Problem with python command : \n {} \n".format(error_py))
                        return(0)

                try:
                    with open(optimization_filename, 'w') as OPATH:
                        OPATH.writelines(command)
                    returncode = subprocess.call(f'chmod +x ./{optimization_filename}',shell=True)
                    if returncode == 0:
                        returncode = subprocess.call(f'./{optimization_filename}',shell=True)
                    if returncode == 1:
                        command = ""
                        command = f'import subprocess \ncommand = "{command}" \nprocess = subprocess.Popen(command.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE) \noutput,error = process.communicate() \nprint("Output : " + str(output) + "\\n Error: " + str(error) + "\\n")'
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
            self.myfitteddata=TDS.myfitdata(xopt)
            self.refreshAll3(message)
        else:
            self.refreshAll3("Output : \n {} \n".format(output))
            print("System not supported. \n")
            print(f'Output : \n {output} \n Error : \n {error} \n')
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

    def name_file(self,path):
        result = ""
        l = len(path)
        for i in range(1,l+1):
            if path[-i]!="/":
                result = result + path[-i]
            else:
                break
        return(result[::-1])

    def preview(self):
        if self.is_temp_file_2 == 1:
            f=open(os.path.join("temp",'temp_file_2.bin'),'rb')
            mesparam=pickle.load(f)
            f.close()
            drudeinput=mesparam[:, 0]
            if self.zvariable==0:
                drudeinput=np.append([self.z],drudeinput)
            self.previewdata=TDS.myfitdata(drudeinput)
            self.refreshAll3("Done")
        else:
            self.refreshAll3("You need to run the 'model parameters' window first.")

    def no_temp_file_1(self):
        self.refreshAll3("Unable to execute without running step Initialisation first.")

    def no_temp_file_2(self):
        self.refreshAll3("Unable to execute without running step 'model parameters' window first.")

    def no_temp_file_4(self):
        self.refreshAll3("Unable to execute without selecting path for output data first.")

    def init(self):
        self.refreshAll("Initialisation: Ok")
