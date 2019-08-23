#!/usr/bin/python
# -*- coding: latin-1 -*-

## This two lines is to chose the econding
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math
from pyswarm import pso   ## Library for optimization
import random
import numpy as np   ## Library to simplify the linear algebra calculations
import scipy.optimize as optimize  ## Library for optimization
import matplotlib.pyplot as plt ## Library for plotting results
from scipy.optimize import curve_fit ## Library for optimization
try:
    import Tkinter as tk  ## Library for GUI python 2
except:
    import tkinter as tk ## Python 3
try:
    import tkFileDialog # For python 2
except:
    import tkinter.filedialog as tkFileDialog # For python 3
    from io import BytesIO
from finalplot import plotall ## Library for plotting results
from finalplot import plotinput  ## Library for plotting results
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
    if myrank == 0:
        print("Number of process used: " + str (size) + '\n')
except: 
    raise ImportError('mpi4py is required for parallelization') 


#end
# =============================================================================
# Extension modules
# =============================================================================

from pyOpt import Optimization   ## Library for optimization
from pyOpt import ALPSO  ## Library for optimization
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
   def __init__(self, pulse,inputSpulse,z):
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
#      self.Spulse= (fft_gpu((calculedpulse(xopt))))
      self.epsilon= Drude(xopt,0)
      self.epsilon_scat = Drude(xopt,1)

###############################################################################
# Graphical User Interface

class AutoScrollbar(tk.Scrollbar): ## https://stackoverflow.com/questions/1873575/how-could-i-get-a-frame-with-a-scrollbar-in-tkinter
    # A scrollbar that hides itself if it's not needed.
    # Only works if you use the grid geometry manager!
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        tk.Scrollbar.set(self, lo, hi)

class ThicknessButton:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Thickness information")
        
        self.vscrollbar = AutoScrollbar(self.root)
        self.vscrollbar.grid(row=0, column=1, sticky="N"+"S")
        self.hscrollbar = AutoScrollbar(self.root, orient="horizontal")
        self.hscrollbar.grid(row=1, column=0, sticky="E"+"W")
        
        self.canvas = tk.Canvas(self.root, yscrollcommand=self.vscrollbar.set,  xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="N"+"S"+"E"+"W")
        
        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)
        
        # make the canvas expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # create canvas contents
        self.window = tk.Frame(self.canvas)
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(1, weight=1)
        
        self.Thick_label = tk.Label(self.window, text = "Tickness of the sample (in m)")
        self.Thick_label.grid(row = 0, column = 0)
        self.Thick = tk.Entry(self.window)
        self.Thick.grid(row = 0, column = 1)
        
        self.label0 = tk.Label(self.window, text = "")
        self.label0.grid(row = 1, column = 1)
        
        self.delta_thick_label = tk.Label(self.window, text = "Uncertainty of the thickness (in %)")
        self.delta_thick_label.grid(row = 2, column = 0)
        self.delta_thick = tk.Entry(self.window)
        self.delta_thick.grid(row = 2, column = 1)
        
        self.label1 = tk.Label(self.window, text = "")
        self.label1.grid(row = 3, column = 1)
        
        self.browse_label1 = tk.Label(self.window, text = "Path of the file without sample")
        self.browse_label1.grid(row = 4, column = 0)
        self.browse_btn1 = tk.Button(self.window, text = "...", command = self.get_pathwithoutsample)
        self.browse_btn1.grid(row = 4, column = 1)
        
        self.label2 = tk.Label(self.window, text = "")
        self.label2.grid(row = 5, column = 1)
        
        self.browse_label2 = tk.Label(self.window, text = "Path of the file with sample")
        self.browse_label2.grid(row = 6, column = 0)
        self.browse_btn2 = tk.Button(self.window, text = "...", command = self.get_pathwithsample)
        self.browse_btn2.grid(row = 6, column = 1)

        
        self.label3 = tk.Label(self.window, text = "")
        self.label3.grid(row = 7, column = 1)
    
        self.submit_btn = tk.Button(self.window, text = "Submit", command = self.get_thickness)
        self.submit_btn.grid(row = 8, column = 1)
        
        self.Thick.bind("<Return>",self.get_thickness)
        self.delta_thick.bind("<Return>",self.get_thickness)
        
        self.canvas.create_window(0, 0, anchor="nw", window=self.window)
        self.window.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.window.mainloop()
        
    def get_thickness(self,event=None): # event is because when we bind the entry case with the event pressing <Return>, this event is consider an entry of the function
        global z, deltaz
        test_variable = 1
        try:
            z = float(self.Thick.get())
            deltaz = float(self.delta_thick.get())
            if z<0 or deltaz>100 or deltaz<0:
                test_variable=0
                self.window2=tk.Tk()
                self.window2.title("Error")
                self.error_label = tk.Label(self.window2,text="Please be sure that the thickness is positive and that the \n uncertainty is between 0 and 100.")
                self.error_label.grid(row=0,column=0)
                self.cancel_btn = tk.Button(self.window2, text = "Ok", command = self.cancel)
                self.cancel_btn.grid(row = 1, column = 0)
                self.window2.mainloop()
        except:
            test_variable=0
            self.window2=tk.Tk()
            self.window2.title("Error")
            self.error_label = tk.Label(self.window2,text="The thickness and the uncertainty must be float or int.")
            self.error_label.grid(row=0,column=0)
            self.cancel_btn = tk.Button(self.window2, text = "Ok", command = self.cancel)
            self.cancel_btn.grid(row = 1, column = 0)
            self.window2.mainloop()
        if test_variable:
            self.root.destroy()
            
    
    def get_pathwithoutsample(self):
        global pathwithoutsample
        self.root1 = tk.Tk()
        self.root1.title("Path without sample")
        pathwithoutsample=tkFileDialog.askopenfilename(parent=self.root1)
        name_filewithoutsample=self.name_file(pathwithoutsample)
        self.browse_btn1.tk.call("grid","remove",self.browse_btn1)
        self.browse_btn1 = tk.Button(self.window, text = name_filewithoutsample, command = self.get_pathwithoutsample)
        self.browse_btn1.grid(row = 4, column = 1)
        self.root1.destroy()
    
    def get_pathwithsample(self):
        global pathwithsample
        self.root2 = tk.Tk()
        self.root2.title("Path with sample")
        pathwithsample=tkFileDialog.askopenfilename(parent=self.root2)
        name_filewithsample=self.name_file(pathwithsample)
        self.browse_btn2.tk.call("grid","remove",self.browse_btn2)
        self.browse_btn2 = tk.Button(self.window, text = name_filewithsample, command = self.get_pathwithsample)
        self.browse_btn2.grid(row = 6, column = 1)
        self.root2.destroy()       
    
    def name_file(self,path):
        result = ""
        l = len(path)
        for i in range(1,l+1):
            if path[-i]!="/":
                result = result + path[-i]
            else:
                break
        return(result[::-1])
    
    def cancel(self):
        self.window2.destroy()



class simulation_choices:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("950x400")
        self.root.title("Simulation model parameters")
        
        self.vscrollbar = AutoScrollbar(self.root)
        self.vscrollbar.grid(row=0, column=1, sticky="N"+"S")
        self.hscrollbar = AutoScrollbar(self.root, orient="horizontal")
        self.hscrollbar.grid(row=1, column=0, sticky="E"+"W")
        
        self.canvas = tk.Canvas(self.root, yscrollcommand=self.vscrollbar.set,  xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="N"+"S"+"E"+"W")
        
        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)
        
        # make the canvas expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # create canvas contents
        self.window = tk.Frame(self.canvas)
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(1, weight=1)
        
        self.choices_list = [tk.StringVar(self.window),tk.StringVar(self.window),tk.StringVar(self.window),
                             tk.StringVar(self.window),tk.StringVar(self.window)] # This list is build as follow [algo choice,model struct,fit thicknes,scattering choice, drude choice, prompt choice]
        self.options_list = [['-','NumPy optimize swarm particle', 'ALPSO without parallelization','ALPSO with parallelization'],
                             [ '-','Transmission Fabry-Perot','Transmission Fabry-Perot \n with a resonator (TDCMT)'],
                             ['-','Yes','No'],['-','Yes','No'],[ '-','Use a file','Do it manually']]
        self.label_text = ["Choose an algorithm \n","Wich model do you whant to use for the photonic structure ? \n",
                           "Is the thickness a variable for the fit ? \n",
                           "Drude model depicts the permitivity Epsillon as Eps =Eps_0- Omega_p^2/(Omega^2-j*gamma*omega) \n Do you want to have a Drude term in the model ? \n","Do you want to use a file with the parameters or do you want to enter them manually ? \n"]
        self.functions = [self.change1,self.change2,self.change3,self.change4,self.change5]
        self.my_labels=[]
        self.pop_menus=[]
        self.l = len(self.choices_list)
        for i in range(self.l):
            if i < self.l-1:
                self.choices_list[i].set(self.options_list[i][0])
                self.my_labels.append(tk.Label(self.window,text=self.label_text[i]))
                self.my_labels[i].grid(row=i,column=0)
                self.pop_menus.append(tk.OptionMenu(self.window,self.choices_list[i],*self.options_list[i]))
                self.pop_menus[i].grid(row=i,column=2)
                self.choices_list[i].trace('w',self.functions[i])
            else:
                self.choices_list[i].set(self.options_list[i][0])
                self.my_labels.append(tk.Label(self.window,text=self.label_text[i]))
                self.my_labels[i].grid(row=self.l+2,column=0)
                self.pop_menus.append(tk.OptionMenu(self.window,self.choices_list[i],*self.options_list[i]))
                self.pop_menus[i].grid(row=self.l+2,column=2)
                self.choices_list[i].trace('w',self.functions[i])

        ######################################################################
        
        self.nb_oscillators_label_1 = tk.Label(self.window, text = "")
        self.nb_oscillators_label_1.grid(row = self.l-1, column = 0)
        self.nb_oscillators_label_2 = tk.Label(self.window, text = "Lorentz model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon*Omega_0^2]/[Omega_0^2+j*gamma*Omega-Omega^2]")
        self.nb_oscillators_label_2.grid(row = self.l, column = 0)
        self.nb_oscillators_label_3 = tk.Label(self.window, text = "Enter a value for the number of Lorentz oscillators in the model:")
        self.nb_oscillators_label_3.grid(row = self.l+1, column = 0)
        self.nb_oscillators = tk.Entry(self.window)
        self.nb_oscillators.grid(row = self.l+1, column = 2)
        
        ######################################################################
        
        self.submit_btn = tk.Button(self.window, text = "Submit", command = self.get_nb_oscillators)
        self.submit_btn.grid(row = self.l+3, column = 2)
        
        self.nb_oscillators.bind("<Return>", self.get_nb_oscillators)
        
        self.canvas.create_window(0, 0, anchor="nw", window=self.window)
        self.window.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.window.mainloop()
   
    
    def change1(self,*args):
        global algo
        algo = self.choices_list[0].get()
        l = len(self.options_list[0])
        for i in range(l):
            if algo == self.options_list[0][i]:
                algo = i
    
    def change2(self,*args):
        global mymodelstruct
        mymodelstruct = self.choices_list[1].get()
        l = len(self.options_list[1])
        for i in range(l):
            if mymodelstruct==self.options_list[1][i]:
                mymodelstruct = i
    
    def change3(self,*args):
        global zvariable
        zvariable = self.choices_list[2].get()
        if zvariable == 'Yes':
            zvariable = 1
        elif zvariable == 'No':
            zvariable = 0

    
    def change4(self,*args):
        global isdrude
        isdrude = self.choices_list[3].get()
        if isdrude == 'Yes':
            isdrude = 1
        elif isdrude == 'No':
            isdrude = 0
    
    def change5(self,*args):
        global prompt
        prompt = self.choices_list[4].get()
        if prompt == self.options_list[4][1]:
            prompt = 0
        elif prompt == self.options_list[4][2]:
            prompt = 1
    
    def get_nb_oscillators(self,event=None):
        global n
        test_variable=1
        error_label=[]
        try:
            n = float(self.nb_oscillators.get())
            if n!=int(n) or n<0:
                error_label = error_label + ["The number of oscillators must be a positive integer"]
                test_variable=0
        except:
            error_label = error_label + ["The number of oscillators must be a float or int"]
            test_variable=0
        for i in range(self.l):
            if self.choices_list[i].get()=='-':
                test_variable=0
                error_label = error_label + ["Please choose an option in line " + str(i+1)]
        if test_variable:
            self.root.destroy()
        else:
            self.window2=tk.Tk()
            self.window2.title("Error")
            nb_errors=len(error_label)
            my_errors=[]
            for i in range(nb_errors):
                my_errors.append(tk.Label(self.window2, text = error_label[i]))
                my_errors[i].grid(row = i+1, column = 0)
            self.cancel_btn = tk.Button(self.window2, text = "Ok", command = self.cancel)
            self.cancel_btn.grid(row = nb_errors + 1, column = 0)
                
        
    def cancel(self):
        self.window2.destroy()


class Input_param:
    global drudeinput,lb,up, nb_param, my_labels, my_values, my_mins, my_maxs, mydescription, myvariables, myunits,mesparam
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1300x400")
        self.root.title("Input parameters")
        
        self.vscrollbar = AutoScrollbar(self.root)
        self.vscrollbar.grid(row=0, column=1, sticky="N"+"S")
        self.hscrollbar = AutoScrollbar(self.root, orient="horizontal")
        self.hscrollbar.grid(row=1, column=0, sticky="E"+"W")
        
        self.canvas = tk.Canvas(self.root, yscrollcommand=self.vscrollbar.set,  xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="N"+"S"+"E"+"W")
        
        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)
        
        # make the canvas expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # create canvas contents
        self.window = tk.Frame(self.canvas)
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(1, weight=1)

        self.param=[]
        for i in range(nb_param):
            label1 = tk.Label(self.window, text="Value")
            label1.grid(row = 0 ,column =1)
            label2 = tk.Label(self.window, text="Min")
            label2.grid(row =0 ,column =2)
            label3 = tk.Label(self.window, text="Max")
            label3.grid(row =0 ,column =3)
            my_labels.append(tk.Label(self.window, text = "\nEnter a value for "+ str(mydescription[i]) + " ("+ str(myvariables[i]) + " in " + str(myunits[i]) + ") :"))
            my_labels[i].grid(row = i+1, column = 0)
            
            self.param.append(tk.StringVar(self.window, value=str(mesparam[i,0])))
            self.param.append(tk.StringVar(self.window, value=str(mesparam[i,1])))
            self.param.append(tk.StringVar(self.window, value=str(mesparam[i,2])))
            
            my_values.append(tk.Entry(self.window,textvariable=self.param[3*i]))
            my_values[i].grid(row = i+1, column = 1)
            my_mins.append(tk.Entry(self.window,textvariable=self.param[1+3*i]))
            my_mins[i].grid(row = i+1, column = 2)
            my_maxs.append(tk.Entry(self.window,textvariable=self.param[2+3*i]))
            my_maxs[i].grid(row = i+1, column = 3)
            
            my_values[i].bind("<Return>",self.my_function)
            my_mins[i].bind("<Return>",self.my_function)
            my_maxs[i].bind("<Return>",self.my_function)
        
    
        self.text_btn = tk.Button(self.window, text = "Submit", command = self.my_function)
        self.text_btn.grid(row = nb_param+1, column = 3)
        
        self.canvas.create_window(0, 0, anchor="nw", window=self.window)
        self.window.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.window.mainloop()
                
    def my_function(self,event=None):
        test_var=1
        error=[]
        for i in range(nb_param):
            try:
                drudeinput[i] = float(my_values[i].get())
                lb[i] = float(my_mins[i].get())
                up[i] = float(my_maxs[i].get())
                if up[i]<=lb[i]:
                    error=error+["The maximum of "+ str(myvariables[i]) +" must be strictly higher than its minimum"]
                    test_var = 0
                elif up[i]<drudeinput[i]:
                    error=error+["The maximum of "+ str(myvariables[i]) +" must be higher than its value"]
                    test_var = 0
                elif lb[i]>drudeinput[i]:
                    error=error+["The minimum of "+ str(myvariables[i]) +" must be lower than its value"]
                    test_var = 0
            except:
                error = error + ["You must enter int or float type"]
                test_var = 0
        
        if test_var:
            self.root.destroy()
        else:
            self.window2=tk.Tk()
            self.window2.title("Error")
            nb_errors=len(error)
            my_errors=[]
            for i in range(nb_errors):
                my_errors.append(tk.Label(self.window2, text = error[i]))
                my_errors[i].grid(row = i+1, column = 0)
            self.cancel_btn = tk.Button(self.window2, text = "Ok", command = self.cancel)
            self.cancel_btn.grid(row = nb_errors + 1, column = 0)
                
    
    def cancel(self):
        self.window2.destroy()


class Continue_opt:
    
    def __init__(self):
        global pathwithoutsample,pathwithsample,z,deltaz
  
        self.window = tk.Tk()
        self.window.title("Optimization")      
        
        self.nb_exp_label = tk.Label(self.window, text = "Do you want to change the parameters value \n and do the optimization again.")
        self.nb_exp_label.grid(row = 0, column = 0)


        self.yes_btn = tk.Button(self.window, text = "Yes", command = self.continue_fnc)
        self.yes_btn.grid(row =1, column = 0)
        
        self.no_btn = tk.Button(self.window, text = "No", command = self.stop_fnc)
        self.no_btn.grid(row =2, column = 0)
                
        self.window.update_idletasks()
        
        self.window.mainloop()
        
    
    
    def continue_fnc(self,event=None):
        global iter_opt
        iter_opt = 1
        
        self.nb_oscillators_label_2 = tk.Label(self.window, text = "Enter the number of Lorentz oscillators")
        self.nb_oscillators_label_2.grid(row = 3, column = 0)
        self.nb_oscillators = tk.Entry(self.window,textvariable=tk.StringVar(self.window, value=str(0)))
        self.nb_oscillators.grid(row = 3, column = 1)
        
        self.submit_btn = tk.Button(self.window, text = "Submit", command = self.get_nb_oscillators)
        self.submit_btn.grid(row = 4, column = 1)
    
    def stop_fnc(self,event=None):
        global iter_opt
        iter_opt = 0
        self.window.destroy()
    
    def get_nb_oscillators(self,event=None):
        global n
        test_variable=1
        error_label=[]
        try:
            n = float(self.nb_oscillators.get())
            if n!=int(n) or n<0:
                error_label = error_label + ["The number of oscillators must be a positive integer"]
                test_variable=0
        except:
            error_label = error_label + ["The number of oscillators must be a float or int"]
            test_variable=0
        if test_variable:
            n=int(n)
            self.window.destroy()
        else:
            self.window2=tk.Tk()
            self.window2.title("Error")
            nb_errors=len(error_label)
            my_errors=[]
            for i in range(nb_errors):
                my_errors.append(tk.Label(self.window2, text = error_label[i]))
                my_errors[i].grid(row = i+1, column = 0)
            self.cancel_btn = tk.Button(self.window2, text = "Ok", command = self.cancel)
            self.cancel_btn.grid(row = nb_errors + 1, column = 0) 



##############################################################################
#  Here one can put an additional refractive index model like Debye for ex.
##############################################################################
            
def Drude(drudeinput,var_int):
    global  myglobalparameters,n,zvariable,mymodelstruct,isdrude
    if var_int==1:
        interm=0
        if zvariable==1:
            interm=interm+1	
        if mymodelstruct==2:
            interm=interm+5
        
        
        eps_inf =drudeinput[0+interm]
        eps =eps_inf*np.ones(len(myglobalparameters.w))
        
        if isdrude==1:	 ## Drude term
            #metaldrude
            wp=drudeinput[1+interm]
            gamma =drudeinput[2+interm]
            eps =eps- wp**2/(1E0+myglobalparameters.w**2-j*gamma* myglobalparameters.w)		
            interm= interm+2
        
        for i in range(0,n):  ## Lorentz term
            chi=drudeinput[i*3+1+interm]
            w0=drudeinput[i*3+2+interm]*2*np.pi
            gamma =drudeinput[i*3+3+interm]*2*np.pi 
            eps =eps+ chi*w0**2/(w0**2+j*gamma* myglobalparameters.w- myglobalparameters.w**2)
    
    else:
        interm=0
        if zvariable==1:
            interm=interm+1	
        if mymodelstruct==2:
            interm=interm+5
        
        eps_inf =drudeinput[0+interm]
        eps =eps_inf*np.ones(len(myglobalparameters.w))
        
        if isdrude==1:	 ## Drude term
            #metaldrude
            wp=drudeinput[1+interm]
            gamma =drudeinput[2+interm]
            eps =eps- wp**2/(1E0+myglobalparameters.w**2-j*gamma* myglobalparameters.w)
            interm= interm+2
        
        for i in range(0,n):  ## Lorentz term
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
	interm=0 ## Intermediate variable use to know exactly the position of the parameters in drudeinput
#first question: is thickness a parameter for optimization?
	if zvariable==1: ## In case we take into account the thickness as a optimization parameter
		interm=interm+1  ## We move by one because now we know the first element of drudeinput is the thickness
		thickn=drudeinput[0]
	else:
		thickn=z
#second question: will we use the TDCMT model or not? 
#ref for TDCMT: https://ieeexplore.ieee.org/document/784592/ 
	if mymodelstruct==2: ## In case is a metamaterial
		w0=drudeinput[0+interm]
		tau0=drudeinput[interm+1]
		tau1=drudeinput[interm+2]
		tau2=drudeinput[interm+3]
		deltatheta=drudeinput[interm+4]
		taue=2/((1/tau1)+(1/tau2))
#		drudeinputbis=np.zeros(len(drudeinput)-5-interm)
		interm=interm+5	## We move by 5 because now we know
        

    
	ref_index=np.sqrt(Drude(np.array(drudeinput),1))
    
    
    #caculation of all the transmission and reflection coefficients
	t12=2/(1+ref_index)  ## Coefficients where 1 is the air and 2 the metamaterial at normal incidence
	t21=2*ref_index/(1+ref_index)
	r22=(ref_index-1)/(1+ref_index)
	r22b=r22

	if mymodelstruct==2: ## In case it is a metamaterial
		deltaw=myglobalparameters.w-w0
		interm1=1/((j*deltaw)+(1/taue)+(1/tau0))
		t12=t12-(1/tau1)*interm1
		r22b=r22b-np.exp(-j*deltatheta)*interm1/np.sqrt(tau1*tau2)
    
    
## In case we have just two interfaces
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
    alpso_none.setOption('filename',"test.out")
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


# =============================================================================
# retrieval of the data
# =============================================================================
    
if myrank == 0:
    # We ask for the thickness information
    z=deltaz=pathwithoutsample=pathwithsample=None
    Thick_btn = ThicknessButton()
    z = float(z)
    deltaz = float(deltaz) / 100.

if myrank != 0:
    z=pathwithsample=pathwithoutsample=None
    
z = comm.bcast(z, root=0)
pathwithsample = comm.bcast(pathwithsample, root=0)
pathwithoutsample = comm.bcast(pathwithoutsample, root=0)

mesdata=np.loadtxt(pathwithsample)    ## We load the signal of the measured pulse with sample
myinputdatafromfile=inputdatafromfile
myglobalparameters=globalparameters
myinputdatafromfile.PulseInittotal=np.loadtxt(pathwithoutsample) ## We load the data of the measured reference pulse
myglobalparameters.t=myinputdatafromfile.PulseInittotal[:,0]*1e-12 #this assumes input files are in ps ## We load the list with the time of the experiment
nsample=len(myglobalparameters.t)
myinputdatafromfile.Pulseinit=myinputdatafromfile.PulseInittotal[:,1]
dt=myglobalparameters.t.item(2)-myglobalparameters.t.item(1)   ## Sample rate 
myglobalparameters.freq = np.fft.rfftfreq(nsample, dt)        ## We create a list with the frequencies for the spectrum
myglobalparameters.w=myglobalparameters.freq*2*np.pi
myinputdatafromfile.Spulseinit=(np.fft.rfft((myinputdatafromfile.Pulseinit)))  ## We compute the spectrum of the measured reference pulse
#    myinputdatafromfile.Spulseinit=(fft_gpu((myinputdatafromfile.Pulseinit)))

if myrank ==0:
    myinputdata=mydata(mesdata[:,1],myinputdatafromfile.Spulseinit,z)    ## We create a variable containing the data related to the measured pulse with sample
    monepsilon=dielcal(np.fft.rfft((mesdata[:,1]))/myinputdatafromfile.Spulseinit,z,myglobalparameters) ## We search for the dielectric function using what we measured
#    monepsilon=dielcal(fft_gpu((mesdata[:,1]))/myinputdatafromfile.Spulseinit,z,myglobalparameters)
    
    ##############################################################################
    #parameters for the algorithm 
    ##############################################################################
    """parameters for the optimization algorithm"""
    swarmsize=1000
    maxiter=20
    
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
    deltaTTT=myglobalparameters.t[np.argmin(myinputdata.pulse)]-myglobalparameters.t[np.argmin(myinputdatafromfile.Pulseinit)] ## retard entre les deux min
    print("Delay between the two minima of the pulses:")
    print("delta T="+ str(deltaTTT))
    print("n="+ str(1+deltaTTT*c/z)) #indice qui en derive
    print("epsillon="+ str(np.square(1+deltaTTT*c/z))) #indice qui en derive
    print("############################################################################################")
    print("############################################################################################")
    deltaTT=np.sum(np.square(myinputdata.pulse)*myglobalparameters.t)/np.sum(np.square(myinputdata.pulse))-np.sum(np.square		(myinputdatafromfile.Pulseinit)*myglobalparameters.t)/np.sum(np.square(myinputdatafromfile.Pulseinit))   #retard entre les deux barycentre, attention pour que ca fonctionne il faut que le rapport signal bruit soit le meme dans les deux cas !!
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
    #we may add here the choice between Drude Lorentz and Debye

    algo=mymodelstruct=zvariable=isdrude=n=prompt=None
    sim_choices = simulation_choices()
    n = int(n)
    
    ###############################################################################
    #Input questions
    ###############################################################################
iter_opt=1
while iter_opt:
    if myrank == 0:
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
        myunits=myunits+["usual permitivity unit without dimension (square of a refractive index)"]
        mydescription=mydescription+["the permitivity at very high frequency frequency"]
        if isdrude==1:
            myvariables=myvariables+["Omega_p"]+["gamma"]
            myunits=myunits+["radian/s"]+["radian/s"]
            mydescription=mydescription+[" the drude Model Plasma frequency : [ (N * (q^2))  /  (Epsillon_0  * m_e) ]"]+["the drude damping rate"]
        for i in range(0,n):
            myvariables=myvariables+["Delta_Epsillon_"+str(i), "1/(2pi)*Omega0_" +str(i),"1/(2pi)*Gamma_"+str(i) ] 
            myunits=myunits+["usual permitivity unit without dimension (square of a refractive index)", "Hz","Hz" ]
            mydescription=mydescription+["the oscillator strentgh of the mode # ", "the frequency of the mode # " +str(i),"the linewidth of the mode # "+str(i) ] 
        drudeinput=np.ones(len(myvariables))
        lb=np.ones(len(myvariables))   ## Array with  the min value of the parameters of the model
        up=np.ones(len(myvariables))   ## Array with  the max value of the parameters of the model
    

        nb_param = len(myvariables)
        my_labels = []
        my_values = []
        my_mins = []
        my_maxs = []
        if prompt:
            mesparam = np.zeros([nb_param,3])
            Input_param()
                
        else: ## This is in case the user doesn't want to put all the parameters of the model manually
            root =tk.Tk()
            print("\nPlease choose the file where all the parameters for the model are\n")
            pathparam=tkFileDialog.askopenfilename(parent=root)
            mesparam=np.loadtxt(pathparam)
            drudeinput=mesparam[:, 0]
            lb=mesparam[:, 1]
            up=mesparam[:, 2]
            root.destroy()
            Input_param()
      
        
    #    =============================================================================
        interm=0  #this is not only in case of a simulated sample
        
        if zvariable==1: ## We take into account the thickness as an optimization parameter so we put the value of the tickness and its uncertainty in the corresponding list
        	drudeinput=np.append([z],drudeinput)
        	lb=np.append([z*(1-deltaz)],lb)
        	up=np.append([z*(1+deltaz)],up)
        	interm=interm+1                
        if mymodelstruct==2:              #if one use resonator tdcmt 
            interm=interm+5
    
                               
        
        # =============================================================================
        # preparation of the input for the algorithm
        # =============================================================================
        print("############################################################################################")
        print("############################################################################################")
        print("begining the calculation please wait ...")
        print("############################################################################################")
        print("############################################################################################")
        
        # Instantiate Optimization Problem 
    
    
    
    if myrank != 0:
        algo=lb=up=drudeinput=swarmsize=maxiter=zvariable=mymodelstruct=isdrude=n=myinputdata=z=pathwithoutsample=pathwithsample=None
    
    ## We broadcast the variables from the master node to the other nodes
    algo = comm.bcast(algo, root=0)
    lb = comm.bcast(lb, root=0)   
    drudeinput = comm.bcast(drudeinput, root=0) 
    up = comm.bcast(up, root=0) 
    swarmsize= comm.bcast(swarmsize, root=0)
    maxiter= comm.bcast(maxiter, root=0)
    zvariable = comm.bcast(zvariable, root=0)
    mymodelstruct = comm.bcast(mymodelstruct, root=0)
    isdrude = comm.bcast(isdrude, root=0)
    n = comm.bcast(n, root=0)
    myinputdata = comm.bcast(myinputdata, root=0) 
    z = comm.bcast(z, root=0)
    pathwithoutsample = comm.bcast(pathwithoutsample, root=0)
    pathwithsample = comm.bcast(pathwithsample, root=0)
    
    ## Optimization dans le cas PyOpt swarm particle ALPSO without parallelization (also works with parallelization)
    if algo>1:
        interm2=0  ## Intermediate variable with a function similar to interm
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
    
    
    if  algo==1: ## xopt is a list we the drudeinput's parameters that minimize 'monerreur', fopt is a list with the optimals objective values
        xopt,fopt=pso(monerreur,lb,up,swarmsize=swarmsize,minfunc=1e-18,minstep=1e-8,debug=1,phip=0.5,phig=0.5,maxiter=maxiter) ## 'monerreur' function that we want to minimize, 'lb' and 'up' bounds of the problem
    
    
    ##############################################################################
    ##############################################################################
    # solving the problem with yopt
    ##############################################################################
    ##############################################################################
    
    if  algo>=2:
        [fopt, xopt, inform] = optimALPSO(opt_prob, swarmsize, maxiter,algo)
    
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
        continue_opt=Continue_opt()
          
    if myrank!=0:
        iter_opt=None
    iter_opt=comm.bcast(iter_opt,root=0)
    

    
if myrank==0:    
    print("\n Please cite this paper in any communication about any use of fit@tds :")
    print("\n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters")
    print("\n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and and Nabil Vindas")
    print("\n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2")
    print("\n DOI: 10.1109/TTHZ.2018.2889227 \n")
    ## Save the data obtained via this program
    print("\n Please choose the file name and path to save the fit results in time domain\n")
    root3=tk.Tk()
    pathoutputime=tkFileDialog.asksaveasfilename()
    fileoutputtime=open(pathoutputime,'wb')
    np.savetxt(fileoutputtime,outputtime,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem Jean-Francois Lampin, Melanie Lavancier and and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n time \t E-field")
    fileoutputtime.close()
    
    outputfreq=abs(np.column_stack((myglobalparameters.freq,myfitteddata.Spulse,np.real(myfitteddata.epsilon),np.imag(myfitteddata.epsilon), np.real(np.sqrt(myfitteddata.epsilon)),np.imag(np.sqrt(myfitteddata.epsilon)),np.real(monepsilon) ,np.imag(monepsilon), np.real(np.sqrt(monepsilon)),np.imag(np.sqrt(monepsilon)) )))
    print("\n Please choose the file name and path to save the fit results in frequency domain\n")
    pathoutpufreq=tkFileDialog.asksaveasfilename()
    fileoutputfreq=open(pathoutpufreq,'wb')
    np.savetxt(fileoutputfreq,outputfreq,header="Please cite this paper in any communication about any use of fit@tds : \n THz-TDS time-trace analysis for the extraction of material and metamaterial parameters \n Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin, Melanie Lavancier and and Nabil Vindas \n IEEE Transactions on Terahertz Science and Technology, Volume 9, Issue 2 \n DOI: 10.1109/TTHZ.2018.2889227 \n \n Freq \t E-field \t real part of fitted epsilon \t imaginary part of fitted epsilon \t real part of fitted n \t imaginary part of fitted n \t real part of initial epsilon \t imaginary part of initial epsilon \t real part of initial n\t imaginary part of initial n")
    fileoutputfreq.close()
    root3.destroy()
    root3.mainloop()
