#!/usr/bin/python
# -*- coding: latin-1 -*-

import os, sys
import pickle
import random
import numpy as np
import traceback
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
try:
    import shutil
except:
    print(traceback.format_exc())
    pass
import h5py

from fit_TDSc import Controler
from epsillon3 import dielcal ## Library for resolving the inverse problem in our case (see the assumptions necessary to use this library)
import fit_TDSf as TDS
from fit_TDSf import Material
from fit_TDSf import Layer
from fit_TDSf import Layers

import fit_TDSm as Model
import csts as Csts

from pathlib import Path as path_
#import optimization as optim


ROOT_DIR = path_(__file__).parent

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
    print(traceback.format_exc())
    print('mpi4py is required for parallelization')
    myrank=0


import sip

def deleteLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                deleteLayout(item.layout())
        sip.delete(layout)

def to_sup(s):
    """Convert a string of digit or integer to superscript"""
    sups = {u'0': u'\u2070',
            u'1': u'\u00b9',
            u'2': u'\u00b2',
            u'3': u'\u00b3',
            u'4': u'\u2074',
            u'5': u'\u2075',
            u'6': u'\u2076',
            u'7': u'\u2077',
            u'8': u'\u2078',
            u'9': u'\u2079'}

    return ''.join(sups.get(char, char) for char in str(s))

graph_option='Transmission'
graph_option_2=None



class MyTableWidget(QWidget):

    def __init__(self, parent,controler):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tab0 = Creation_tab(self,controler)
        self.tab1 = Initialisation_tab(self,controler)
        self.tab2 = Model_parameters_tab(self,controler)
        self.tab3 = Optimization_tab(self,controler)

        # Add tabs
        self.tabs.addTab(self.tab0,"Create material")
        self.tabs.addTab(self.tab1,"Initialisation")
        self.tabs.addTab(self.tab2,"Model parameters")
        self.tabs.addTab(self.tab3,"Optimization")
        self.tabs.setCurrentIndex(1)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

###############################################################################
###############################################################################
##########################   Creation tab   ###################################
###############################################################################
###############################################################################

class Creation_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.controler = controler
        
        self.category_choice = QComboBox()
        self.category_choice.addItems(['Material','Metasurface'])
        self.category_choice.setMaximumWidth(440)
        self.category_choice.currentIndexChanged.connect(self.refresh)
        self.material_choices = material_choices_handler(self,controler)
        self.material_choices.refresh()
        self.material_param = material_parameters(self,controler)
        self.log_box = log_material_choices(self,controler)
        self.graphs = Graphs_Creation(self,controler)
        
        # Creation Layouts
        main_layout = QHBoxLayout()
        sub_layout_v1 = QVBoxLayout()
        sub_layout_v2 = QVBoxLayout()
        
        # Organisation layouts
        sub_layout_v1.addWidget(self.category_choice,0)
        sub_layout_v1.addWidget(self.material_choices)
        sub_layout_v1.addWidget(self.log_box)
        sub_layout_v1.addWidget(self.material_param)
        sub_layout_v2.addWidget(self.graphs)

        main_layout.addLayout(sub_layout_v1)
        main_layout.addLayout(sub_layout_v2)
        self.setLayout(main_layout)
        
    def refresh(self): # /!\ Creation_tab is not a client of the controler, 
    #this is not called by the controler, only when category_choice is changed.
        deleteLayout(self.material_param.layout())
        self.controler.nb_param0 = 0
        self.controler.refreshAll0('')
        
class material_choices_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient0(self)

    def refresh(self):
        material_choices_instance = material_choices(self,self.controler,self.parent.category_choice.currentIndex())
 #       if self.parent.category_choice.currentIndex() == 0:
        try:
            self.main_layout=QVBoxLayout()
            self.main_layout.addWidget(material_choices_instance)
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)
            self.setMaximumHeight(450)
        except:
            print(traceback.format_exc())
#        if self.parent.category_choice.currentIndex() == 1:
#            try:
#                deleteLayout(self.layout())
#                self.setMaximumHeight(0)
#            except:
#                print(traceback.format_exc())
#                pass
        
class material_choices(QGroupBox):
    def __init__(self, parent, controler,isInterface):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient0(self)
        self.isInterface=isInterface
        if not self.isInterface:
            self.setTitle("Material choices")
        else:
            self.setTitle("Metasurface choices")
        self.setFixedWidth(430)
        self.setMaximumHeight(450)

        # Creation widgets
        label_width=200
        action_widget_width=200
        corrective_width_factor=0

        self.label_model = []
        self.enter_model = []

        self.main_layout = QVBoxLayout()
        sub_layout_h = []
        if not self.isInterface:                                                                  #Material
            for i in range(len(Model.materialModels)):
                self.label_model.append(QLabel(Model.materialModels[i].label + to_sup(i+1)))
                self.label_model[i].setMaximumWidth(label_width)
                if Model.materialModels[i].isCumulative:
                    self.enter_model.append(QLineEdit())
                    self.enter_model[i].setMaximumWidth(action_widget_width + corrective_width_factor)
                    self.enter_model[i].setMaximumHeight(30)
                    self.enter_model[i].setText("0")
                else:
                    self.enter_model.append(QComboBox())
                    self.enter_model[i].addItems(['No','Yes'])
                    self.enter_model[i].setMaximumWidth(action_widget_width)                   
                # Creation layouts
                sub_layout_h.append(QHBoxLayout())
                # Organisation Layouts
                sub_layout_h[i].addWidget(self.label_model[i],0)
                sub_layout_h[i].addWidget(self.enter_model[i],0)
                self.main_layout.addLayout(sub_layout_h[i])
        else:                                                                               #Metasurface
            for i in range(len(Model.interfaceModels)):
                self.label_model.append(QLabel(Model.interfaceModels[i].label + to_sup(len(Model.materialModels)+i+1)))
                self.label_model[i].setMaximumWidth(label_width)
                self.enter_model.append(QLineEdit())
                self.enter_model[i].setMaximumWidth(action_widget_width + corrective_width_factor)
                self.enter_model[i].setMaximumHeight(30)
                self.enter_model[i].setText("0")
                # Creation layouts
                sub_layout_h.append(QHBoxLayout())
                # Organisation Layouts
                sub_layout_h[i].addWidget(self.label_model[i],0)
                sub_layout_h[i].addWidget(self.enter_model[i],0)
                self.main_layout.addLayout(sub_layout_h[i])
        
        

       
        # OK button
        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(self.submit_model_param)

        self.main_layout.addWidget(self.button_submit)

        self.setLayout(self.main_layout)

    def submit_model_param(self):
        nbTerms = [] 

        if not self.isInterface:                                         #Material
            for j in range(len(Model.materialModels)):
                try:
                    if Model.materialModels[j].isCumulative:
                        nbTerms.append(int(self.enter_model[j].text()))
                    else:
                        nbTerms.append(self.enter_model[j].currentIndex())
                    if nbTerms[j]<0:
                        self.controler.invalid_n_model0(Model.materialModels[j].invalidNumberMessage)
                        return(0)
                except:
                    print(traceback.format_exc())
                    self.controler.invalid_n_model0(Model.materialModels[j].invalidNumberMessage)
                    return(0)
        else:
            for j in range(len(Model.interfaceModels)):
                try:
                    nbTerms.append(int(self.enter_model[j].text()))
                    if nbTerms[j]<0:
                        self.controler.invalid_n_model0("Invalid number of term for the model: {}".format(Model.interfaceModels[j].label))
                        return(0)
                except:
                    print(traceback.format_exc())
                    self.controler.invalid_n_model0("Invalid number of term for the model: {}".format(Model.interfaceModels[j].label))
                    return(0)
        self.controler.material_parameters(nbTerms,self.isInterface)


    def refresh(self):
        pass

class log_material_choices(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.parent = parent
        self.controler.addClient0(self)
        self.setReadOnly(True)
        self.setMaximumWidth(440)
        self.setMaximumHeight(200)
        for j in range(len(Model.materialModels)):
            self.append(to_sup(j+1) + " " + Model.materialModels[j].explanation + "\n")
        for j in range(len(Model.interfaceModels)):
            self.append(to_sup(len(Model.materialModels)+j+1) + " " + Model.interfaceModels[j].explanation + "\n")
    def refresh(self):
        if self.parent.category_choice.currentIndex() == 1:
            self.setMaximumHeight(150)
        else:
            self.setMaximumHeight(200)
        message = self.controler.message
        if message:
            self.append(message)
            
class material_parameters(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient0(self)
        if self.parent.category_choice.currentIndex() == 0:                                     #Material
            self.setTitle("Material parameters")
        else:
            self.setTitle("Metasurface parameters")
        self.setFixedWidth(450)

    def refresh(self):
            
        nb_param=self.controler.nb_param0
        
        material_parameters=material_parameters_scroll(self,self.controler)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(material_parameters)
        
        self.main_layout=QVBoxLayout()
        self.main_layout.addWidget(scroll)
        
        if nb_param == 0:
            deleteLayout(self.layout())
        else:
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)
        

class material_parameters_scroll(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient0(self)

        mydescription=self.controler.mydescription0
        myunits=self.controler.myunits0
        nb_param=self.controler.nb_param0

        # Path of a file containing the values of the parameters (optional)
        self.path_param_values=None
        # Directory and name to save parameters in external file
        self.dir_save_param=None
        self.name_save_parameters=None

        label_width=1500
        text_box_width=100
        text_box_height=25


        # Creation Widgets et Layouts
        self.label_param=QLabel('Parameter')
        self.label_param.setMaximumWidth(label_width)

        self.label_value=QLabel('Value')
        self.label_value.setMaximumWidth(text_box_width)

        self.label_search_path=QLabel('Use file to enter parameters values (optional)')
        self.label_search_path.setMaximumWidth(label_width)

        self.labels=[]

        self.main_layout=QVBoxLayout()
        sub_layout_h=QHBoxLayout()
        sub_layout_h2=QHBoxLayout()
        sub_layout_h3=QHBoxLayout()
        sub_layout_h4=QHBoxLayout()
        sub_layout_h5=QHBoxLayout()
        layouts=[]

        self.text_boxes_value=[]

        self.search_path_button = QPushButton("browse")
        self.search_path_button.clicked.connect(self.search_path)
        self.search_path_button.setMaximumWidth(text_box_width)
        self.search_path_button.setMaximumHeight(text_box_height)

        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.preview_button.setMaximumHeight(text_box_height)
        # Widgets to enter the name of the file to save parameters
        self.label_name = QLabel('Name of the material/metasurface')
        self.label_name.setMaximumWidth(label_width)
        self.enter_name = QLineEdit()
        self.enter_name.setMaximumWidth(text_box_width)
        # Widgets to enter file directory
        self.label_path = QLabel('Path of the parameters file')
        self.label_path.setMaximumWidth(label_width)
        self.search_save_path_button = QPushButton("browse")
        self.search_save_path_button.clicked.connect(self.search_save_path)
        self.search_save_path_button.setMaximumWidth(text_box_width)
        self.search_save_path_button.setMaximumHeight(text_box_height)
        # Button to save parameters in the desired file
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_values)
        self.save_button.setMaximumHeight(text_box_height)
        # Error box
        self.log_box=log_material_param(self,self.controler)
        self.log_box.setMaximumWidth(275)
        self.log_box.setMaximumHeight(30)


        if nb_param:
            for i in range(nb_param):
                self.labels.append(QLabel('{0} ({1})'.format(mydescription[i],myunits[i])))
                self.labels[i].setMaximumWidth(label_width)
                self.labels[i].setAlignment(Qt.AlignTop)

                self.text_boxes_value.append(QLineEdit())
                self.text_boxes_value[i].setMaximumWidth(text_box_width)
                self.text_boxes_value[i].setMaximumHeight(text_box_height)

                layouts.append(QHBoxLayout())
                layouts[i].setAlignment(Qt.AlignTop)

        # Organisation layouts
        sub_layout_h.addWidget(self.label_param)
        sub_layout_h.addWidget(self.label_value)
        sub_layout_h.setAlignment(Qt.AlignTop)
        self.main_layout.addLayout(sub_layout_h)

        if nb_param:
            for i in range(nb_param):
                layouts[i].addWidget(self.labels[i])
                layouts[i].addWidget(self.text_boxes_value[i])
                self.main_layout.addLayout(layouts[i])

        sub_layout_h2.addWidget(self.label_search_path)
        sub_layout_h2.addWidget(self.search_path_button)
        sub_layout_h3.addWidget(self.log_box)
        sub_layout_h3.addWidget(self.preview_button)
        sub_layout_h4.addWidget(self.label_name)
        sub_layout_h4.addWidget(self.enter_name)
        sub_layout_h5.addWidget(self.label_path)
        sub_layout_h5.addWidget(self.search_save_path_button)
        
        self.main_layout.addLayout(sub_layout_h2)
        self.main_layout.addLayout(sub_layout_h3)
        self.main_layout.addLayout(sub_layout_h4)
        self.main_layout.addLayout(sub_layout_h5)
        self.main_layout.addWidget(self.save_button)

        self.setLayout(self.main_layout)

    def refresh(self):
        pass

    def preview(self):
        global myepsilonmaterial, freq0, graph_option_0
        nb_param=self.controler.nb_param0
        mesparam = np.zeros(nb_param)
        try:
            for i in range(nb_param):
                mesparam[i]=float(self.text_boxes_value[i].text())
        except:
            print(traceback.format_exc())
            self.log_box.append("Invalid values.")
            return(0)

        f=open(os.path.join("temp",'temp_file_0.bin'),'rb')      
        nbTerms = pickle.load(f)
        f.close()
        self.controler.material0.change_param(mesparam,self.controler.material0.variableNames())
        try:
            myepsilonmaterial = self.controler.material0.epsilon(2*np.pi*freq0)      
            graph_option_0 ='Real(refractive index)'
            self.controler.refreshAll0('')
        except AttributeError:
            self.log_box.append("Previews are not implemented for metasurfaces")
            
    def search_path(self):
        nb_param=self.controler.nb_param0
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        try:
            name=os.path.basename(fileName)
            self.search_path_button.setText(name)
            self.log_box.append("Values taken from "+name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path0()
            return(0)
        try:
            mes_param=np.loadtxt(fileName,dtype = np.float64)
            if nb_param==len(mes_param):
                for i in range(nb_param):
                    self.text_boxes_value[i].setText('{0:.3E}'.format(mes_param[i]))
            else:
                self.log_box.append("The file submitted does not have the same number of parameters as the model chosen.")
                return(0)
        except:
            print(traceback.format_exc())
            self.log_box.append("There is a problem with the file submitted.")
            return(0)
            
    def search_save_path(self):
        #find path to save parameters
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.dir_save_param=str(DirectoryName)
            name=os.path.basename(str(DirectoryName))
            self.search_save_path_button.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path0()
            
    def save_values(self):
        nb_param=self.controler.nb_param0
        mesparam = np.zeros(nb_param)
        try:
            for i in range(nb_param):
                mesparam[i]=float(self.text_boxes_value[i].text())
        except:
            print(traceback.format_exc())
            self.log_box.append("Invalid values.")
            return(0)
        name = self.enter_name.text()
        if name == '':
            self.log_box.append('Please enter a name')
            return(0)
        if self.dir_save_param == None:
                self.log_box.append("Please enter a valid path")
                return(0)
        if self.parent.parent.category_choice.currentIndex() == 0:
            f=open(os.path.join("temp",'temp_file_0.bin'),'rb')
            nbTerms = pickle.load(f)
            f.close()
            self.controler.material0 = Material(name = name, nbTerms = nbTerms, param = mesparam)
        else:
            self.controler.material0 = TDS.Interface(name = name, isMetasurface = 1, param = mesparam)
        self.controler.save_material_param(self.controler.material0,self.dir_save_param)
        self.log_box.append("Values saved")
            

class log_material_param(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient0(self)
        self.setReadOnly(True)
        self.append('Log')
    def refresh(self):
        pass
    
class Graphs_Creation(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient0(self)
        self.setTitle("Graphs")
        # Create objects to plot graphs
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.draw()
        # Create buttons to chose what to plot
        # Real part of refractive index
        self.button_real_index = QPushButton('Real(n)', self)
        self.button_real_index.clicked.connect(self.real_index_graph)
        # Imaginary part of refractive index
        self.button_im_index = QPushButton('Im(n)', self)
        self.button_im_index.clicked.connect(self.im_index_graph)
        #Permittivity
        self.button_Permittivity = QPushButton('Permittivity', self)
        self.button_Permittivity.clicked.connect(self.Permitivity_graph)
        
        # Organisation layout
        self.vlayoutmain = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.button_real_index)
        self.hlayout.addWidget(self.button_im_index)
        self.hlayout.addWidget(self.button_Permittivity)
        self.vlayoutmain.addWidget(self.canvas)
        self.vlayoutmain.addLayout(self.hlayout)
        self.setLayout(self.vlayoutmain)
    
    def draw_graph_material(self,freq,epsilon):
        global graph_option_0
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        if graph_option_0=='Real(refractive index)':
            ax1.set_title('Real part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Real part of refractive index',color=color)
            ax1.plot(freq, np.sqrt(epsilon).real, 'b-', label='Re(n)')
            ax1.legend()
        elif graph_option_0=='Im(refractive index)':
            ax1.set_title('Imaginary part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Imaginary part of refractive index',color=color)
            ax1.plot(freq, np.sqrt(epsilon).imag, 'r-', label='Im(n)')
            ax1.legend()
        elif graph_option_0=='Permittivity':
            ax1.set_title('Permittivity', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Permittivity',color=color)
            ax1.plot(freq,  np.real(epsilon), 'b-', label='real part')
            ax1.plot(freq, np.imag(epsilon), 'r-', label='imaginary part')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend()
        self.figure.tight_layout()
        self.canvas.draw()
        
    def Permitivity_graph(self):
        global graph_option_0
        graph_option_0='Permittivity'
        self.controler.ploting_text0('Plotting Permittivity')
    
    def real_index_graph(self):
        global graph_option_0
        graph_option_0='Real(refractive index)'
        self.controler.ploting_text0('Plotting real part of refractive index')

    def im_index_graph(self):
        global graph_option_0
        graph_option_0='Im(refractive index)'
        self.controler.ploting_text0('Plotting imaginary part of refractive index')
    
    def refresh(self):
        global myepsilonmaterial,freq0, graph_option_0
        try:        
            freq0=self.controler.myglobalparameters.freq
            if freq0==None:
                freq0 = np.arange(0,5e12,1e9)
            self.draw_graph_material(freq0,myepsilonmaterial)
        except:
            print(traceback.format_exc())
            self.figure.clf()
            ax1 = self.figure.add_subplot(111)
            ax1.set_title('Graphs', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('')
            ax1.set_ylabel('',color=color)
            ax1.plot(0,0,color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            self.figure.tight_layout()
            self.canvas.draw()

###############################################################################
###############################################################################
#######################   Initialisation tab   ################################
###############################################################################
###############################################################################

class Initialisation_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.setMinimumSize(640, 480)
        hlayout  = QHBoxLayout()
        vlayout1 = QVBoxLayout()
        vlayout2 = QVBoxLayout()
        
        self.advanced_choice = QComboBox()
        self.advanced_choice.addItems(['Do not use advanced options','Use advanced options'])
        self.advanced_choice.currentIndexChanged.connect(self.refresh)
        self.advanced_choice.setMaximumWidth(440)
        
        self.init_param_widget = InitParam_handler(self, controler)
        self.init_param_widget.refresh()
        self.text_box = TextBoxWidget(self, controler)
        self.layout_materials = layout_materials(self, controler)
        self.layers = layers(self, controler)
        
        vlayout1.addWidget(self.advanced_choice)
        vlayout1.addWidget(self.init_param_widget, 0)
        vlayout1.addWidget(self.text_box, 0)
        vlayout2.addWidget(self.layout_materials, 0)
        vlayout2.addWidget(self.layers, 0)
        hlayout.addLayout(vlayout1, 0)
        hlayout.addLayout(vlayout2, 1)
        self.setLayout(hlayout)
        
    def refresh(self):# /!\ Initialisation_tab is not a client of the controler, 
    #this is not called by the controler, only when advanced_choice is changed.
        deleteLayout(self.init_param_widget.layout())
        self.controler.nlayers = None
        self.controler.nfixed_material = None
        self.controler.noptim_material = None
        self.controler.nfixed_metasurface = None
        self.controler.noptim_metasurface = None
        self.init_param_widget.refresh()
        self.controler.refreshAll('')
        
class InitParam_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setMaximumWidth(440)

    def refresh(self):
        init_instance = InitParamWidget(self.parent,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except AttributeError:
            main_layout = QVBoxLayout()
            main_layout.addWidget(init_instance)
            self.setLayout(main_layout)


class InitParamWidget(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient(self)
#        self.setTitle("Layout")
        
        label_width=1500
        text_box_width=200
        text_box_height=25
        
        # Number of layers
        self.label_nlayers = QLabel('Number of layers')
        self.label_nlayers.setMaximumWidth(label_width)
        self.nlayers_box=QLineEdit()
        self.nlayers_box.setMaximumWidth(text_box_width)
        self.nlayers_box.setMaximumHeight(text_box_height)
        self.nlayers_box.setText('1')
        # Number of fixed materials
        self.label_nfixed_material = QLabel('Number of fixed materials')
        self.label_nfixed_material.setMaximumWidth(label_width)
        self.nfixed_material_box=QLineEdit()
        self.nfixed_material_box.setAlignment(Qt.AlignVCenter)
        self.nfixed_material_box.setMaximumWidth(text_box_width)
        self.nfixed_material_box.setMaximumHeight(text_box_height)
        self.nfixed_material_box.setText('0')
        # Number of materials to optimize
        self.label_nmaterial_optim = QLabel('Number of materials to optimize')
        self.label_nmaterial_optim.setMaximumWidth(label_width)
        self.nmaterial_optim_box=QLineEdit()
        self.nmaterial_optim_box.setMaximumWidth(text_box_width)
        self.nmaterial_optim_box.setMaximumHeight(text_box_height)
        self.nmaterial_optim_box.setText('1')
        
        #modify for metasurfaces:
        
        # Number of fixed resonators
        self.label_nfixed_metasurface = QLabel('Number of fixed metasurfaces ' + to_sup(1))
        self.label_nfixed_metasurface.setMaximumWidth(label_width)
        self.nfixed_metasurface_box=QLineEdit()
        self.nfixed_metasurface_box.setAlignment(Qt.AlignVCenter)
        self.nfixed_metasurface_box.setMaximumWidth(text_box_width)
        self.nfixed_metasurface_box.setMaximumHeight(text_box_height)
        self.nfixed_metasurface_box.setText('0')
        # Number of resonators to optimize
        self.label_noptim_metasurface = QLabel('Number of metasurfaces to optimize ' + to_sup(1))
        self.label_noptim_metasurface.setMaximumWidth(label_width)
        self.noptim_metasurface_box=QLineEdit()
        self.noptim_metasurface_box.setMaximumWidth(text_box_width)
        self.noptim_metasurface_box.setMaximumHeight(text_box_height)
        self.noptim_metasurface_box.setText('0')
        
        
        # We create the text associated to the text box

        self.label_path_without_sample = QLabel('Select data (without sample) '+to_sup(2))
        self.label_path_without_sample.setAlignment(Qt.AlignVCenter)
        self.label_path_without_sample.resize(200, 100);
        self.label_path_without_sample.resize(self.label_path_without_sample.sizeHint());

        self.label_path_with_sample = QLabel('Select data (with sample) '+to_sup(2))
        self.label_path_with_sample.setAlignment(Qt.AlignVCenter)
        self.label_path_with_sample.resize(200, 100);
        self.label_path_with_sample.resize(self.label_path_with_sample.sizeHint());

        # We create text box for the user to enter values of the sample

        self.button_ask_path_without_sample = QPushButton('browse')
        self.button_ask_path_without_sample.resize(200, 100);
        self.button_ask_path_without_sample.resize(self.button_ask_path_without_sample.sizeHint());
        self.button_ask_path_without_sample.clicked.connect(self.get_path_without_sample)

        self.button_ask_path_with_sample = QPushButton('browse')
        self.button_ask_path_with_sample.resize(200, 100);
        self.button_ask_path_with_sample.resize(self.button_ask_path_with_sample.sizeHint());
        self.button_ask_path_with_sample.clicked.connect(self.get_path_with_sample)

        # We create a button to extract the information from the text boxes
        self.button = QPushButton('Submit')
        self.button.clicked.connect(self.on_click)
        self.button.pressed.connect(self.pressed_loading1)
        
        # Filter or not filter
        self.LFfilter_label = QLabel('Filter low frequencies?')
        self.LFfilter_choice = QComboBox()
        self.LFfilter_choice.addItems(['No','Yes'])
        self.LFfilter_choice.setMaximumWidth(text_box_width)
        self.LFfilter_choice.setMaximumHeight(text_box_height)
        
        self.HFfilter_label = QLabel('Filter high frequencies?')
        self.HFfilter_choice = QComboBox()
        self.HFfilter_choice.addItems(['No','Yes'])
        self.HFfilter_choice.setMaximumWidth(text_box_width)
        self.HFfilter_choice.setMaximumHeight(text_box_height)
        
        self.label_start = QLabel('Start (Hz)')
        self.label_end   = QLabel('End (Hz)')
        self.label_sharp = QLabel('Sharpness of frequency filter '+to_sup(3))
        self.start_box = QLineEdit()
        self.end_box   = QLineEdit()
        self.sharp_box = QLineEdit()
        self.start_box.setMaximumWidth(text_box_width)
        self.start_box.setMaximumHeight(text_box_height)
        self.start_box.setText("0.18e12")
        self.end_box.setMaximumWidth(text_box_width)
        self.end_box.setMaximumHeight(text_box_height)
        self.end_box.setText("6e12")
        self.sharp_box.setMaximumWidth(text_box_width)
        self.sharp_box.setMaximumHeight(text_box_height)
        self.sharp_box.setText("10")
        
        # remove end of reference pulse
        self.label_zeros = QLabel('Set end of time trace to zero? '+to_sup(4))
        self.zeros_choice = QComboBox()
        self.zeros_choice.addItems(['No','Yes'])
        self.zeros_choice.setMaximumWidth(text_box_width)
        self.zeros_choice.setMaximumHeight(text_box_height)
        
        #Remove baseline "dark" noise
        self.label_dark = QLabel('Remove dark noise ramp ? '+to_sup(5))
        self.dark_choice = QComboBox()
        self.dark_choice.addItems(['No','Yes']) #Use a function to add
        self.dark_choice.setMaximumWidth(text_box_width-24)
        
        self.label_slope = QLabel('Slope of the ramp?')
        self.slope_box = QLineEdit()
        self.slope_box.setMaximumWidth(text_box_width)
        self.slope_box.setMaximumHeight(text_box_height)
        self.slope_box.setText("4e-6")
        
        self.label_intercept = QLabel('Intercept of the ramp?')
        self.intercept_box = QLineEdit()
        self.intercept_box.setMaximumWidth(text_box_width)
        self.intercept_box.setMaximumHeight(text_box_height)
        self.intercept_box.setText("0.3e-3")
        
        # Delay
        self.label_delay = QLabel("Fit delay")
        self.label_delay.setMaximumWidth(label_width)
        self.options_delay = QComboBox()
        self.options_delay.addItems(['No','Yes'])
        self.options_delay.setMaximumWidth(text_box_width-24)
        self.delayvalue_label = QLabel("Initial guess, maximum delay (s)")
        self.delay_guess_box = QLineEdit()
        self.delay_guess_box.setMaximumWidth(text_box_width-24)
        self.delay_limit_box = QLineEdit()
        self.delay_limit_box.setMaximumWidth(text_box_width-24)
        self.fix_delay_box = QCheckBox("Fixed")
        
        # Leftover noise
        self.label_leftover = QLabel("Fit the amplitude variation and time scale dilation")
        self.label_leftover.setMaximumWidth(label_width)
        self.options_leftover = QComboBox()
        self.options_leftover.addItems(['No','Yes'])
        self.options_leftover.setMaximumWidth(text_box_width-24)
        self.leftovervaluea_label = QLabel("Amplitude coefficient guess and maximum value")
        self.leftovera_guess_box = QLineEdit()
        self.leftovera_guess_box.setMaximumWidth(text_box_width-24)
        self.leftovera_limit_box = QLineEdit()
        self.leftovera_limit_box.setMaximumWidth(text_box_width-24)
        # self.label_fix_a = QLabel("Fixed")
        # self.label_fix_a.setMaximumWidth(label_width)
        self.fix_a_box = QCheckBox("Fixed")
        #self.fix_a_box.setMaximumWidth(text_box_width-24)
        self.leftovervaluec_label = QLabel("Dilation coefficient guess and maximum value")
        self.leftoverc_guess_box = QLineEdit()
        self.leftoverc_guess_box.setMaximumWidth(text_box_width-24)
        self.leftoverc_limit_box = QLineEdit()
        self.leftoverc_limit_box.setMaximumWidth(text_box_width-24)  
        self.fix_c_box = QCheckBox("Fixed")
        
        # Super resolution
        self.label_super = QLabel("Super resolution")
        self.options_super = QComboBox()
        self.options_super.addItems(['No','Yes'])
        self.options_super.setMaximumWidth(text_box_width-24)
        

        # Organisation layout
        self.hlayout1=QHBoxLayout()
        self.hlayout2=QHBoxLayout()
        self.hlayout3=QHBoxLayout()
        self.hlayout4=QHBoxLayout()
        self.hlayout5=QHBoxLayout()
        self.hlayout6=QHBoxLayout()
        self.hlayout7=QHBoxLayout()
        self.hlayout8=QHBoxLayout()
        self.hlayout9=QHBoxLayout()
        self.hlayout10=QHBoxLayout()
        self.hlayout11=QHBoxLayout()
        self.hlayout12=QHBoxLayout()
        self.hlayout14=QHBoxLayout()
        self.hlayout17=QHBoxLayout()
        self.hlayout18=QHBoxLayout()
        self.hlayout19=QHBoxLayout()
        self.hlayout20=QHBoxLayout()
        self.hlayout21=QHBoxLayout()
        self.hlayout22=QHBoxLayout()
        self.hlayout23=QHBoxLayout()
        self.vlayoutmain=QVBoxLayout()
        
        self.hlayout1.addWidget(self.label_nlayers,1)
        self.hlayout1.addWidget(self.nlayers_box,0)
        
        self.hlayout2.addWidget(self.label_nfixed_material,1)
        self.hlayout2.addWidget(self.nfixed_material_box,0)
        
        self.hlayout3.addWidget(self.label_nmaterial_optim,1)
        self.hlayout3.addWidget(self.nmaterial_optim_box,0)
        
        self.hlayout4.addWidget(self.label_nfixed_metasurface,1) #modify for metasurfaces
        self.hlayout4.addWidget(self.nfixed_metasurface_box,0)
        
        self.hlayout5.addWidget(self.label_noptim_metasurface,1)
        self.hlayout5.addWidget(self.noptim_metasurface_box,0)
        
        self.hlayout6.addWidget(self.label_path_without_sample,20)
        self.hlayout6.addWidget(self.button_ask_path_without_sample,17)

        self.hlayout7.addWidget(self.label_path_with_sample,20)
        self.hlayout7.addWidget(self.button_ask_path_with_sample,17)
        
        self.hlayout8.addWidget(self.label_super,0)
        self.hlayout8.addWidget(self.options_super,1)
        
        self.hlayout9.addWidget(self.label_delay,0)
        self.hlayout9.addWidget(self.options_delay,1)

        self.hlayout10.addWidget(self.delayvalue_label,0)
        self.hlayout10.addWidget(self.delay_guess_box,1)
        self.hlayout10.addWidget(self.delay_limit_box,1)
        self.hlayout10.addWidget(self.fix_delay_box,0)
        
        self.hlayout11.addWidget(self.label_leftover,0)
        self.hlayout11.addWidget(self.options_leftover,1)
        
        self.hlayout12.addWidget(self.leftovervaluea_label,0)
        self.hlayout12.addWidget(self.leftovera_guess_box,1)
        self.hlayout12.addWidget(self.leftovera_limit_box,1)
        self.hlayout12.addWidget(self.fix_a_box,0)
        
        self.hlayout14.addWidget(self.leftovervaluec_label,0)
        self.hlayout14.addWidget(self.leftoverc_guess_box,1)
        self.hlayout14.addWidget(self.leftoverc_limit_box,1)
        self.hlayout14.addWidget(self.fix_c_box,0)
        
        self.hlayout17.addWidget(self.LFfilter_label,1)
        self.hlayout17.addWidget(self.LFfilter_choice,0)
        self.hlayout17.addWidget(self.label_start,1)
        self.hlayout17.addWidget(self.start_box,0)
        
        self.hlayout18.addWidget(self.HFfilter_label,1)
        self.hlayout18.addWidget(self.HFfilter_choice,0)
        self.hlayout18.addWidget(self.label_end,1)
        self.hlayout18.addWidget(self.end_box,0)

        self.hlayout19.addWidget(self.label_sharp,1)
        self.hlayout19.addWidget(self.sharp_box,0)

        self.hlayout20.addWidget(self.label_zeros,1)
        self.hlayout20.addWidget(self.zeros_choice,0)
        
        self.hlayout21.addWidget(self.label_dark,1)
        self.hlayout21.addWidget(self.dark_choice,0)
        self.hlayout22.addWidget(self.label_slope,1)
        self.hlayout22.addWidget(self.slope_box,0)
        self.hlayout23.addWidget(self.label_intercept,1)
        self.hlayout23.addWidget(self.intercept_box,0)

        self.vlayoutmain.addLayout(self.hlayout1)
        self.vlayoutmain.addLayout(self.hlayout2)
        self.vlayoutmain.addLayout(self.hlayout3)
        self.vlayoutmain.addLayout(self.hlayout4) #modify for metasurfaces
        self.vlayoutmain.addLayout(self.hlayout5)
        self.vlayoutmain.addLayout(self.hlayout6)
        self.vlayoutmain.addLayout(self.hlayout7)
        if self.parent.advanced_choice.currentIndex() == 1:
            self.vlayoutmain.addLayout(self.hlayout8)
            self.vlayoutmain.addLayout(self.hlayout9)
            self.vlayoutmain.addLayout(self.hlayout10)
            self.vlayoutmain.addLayout(self.hlayout11)
            self.vlayoutmain.addLayout(self.hlayout12)
            self.vlayoutmain.addLayout(self.hlayout14)
            self.vlayoutmain.addLayout(self.hlayout21)
            self.vlayoutmain.addLayout(self.hlayout22)
            self.vlayoutmain.addLayout(self.hlayout23)
        sub_layoutv = QVBoxLayout()
        sub_layoutv.addLayout(self.hlayout17)
        sub_layoutv.addLayout(self.hlayout18)
        sub_layoutv.addLayout(self.hlayout19)
        sub_layoutv.addLayout(self.hlayout20)
        filter_group = QGroupBox()
        filter_group.setTitle('Filters')
        filter_group.setLayout(sub_layoutv)
        self.vlayoutmain.addWidget(filter_group)
        self.vlayoutmain.addWidget(self.button)
        self.setLayout(self.vlayoutmain)


    def pressed_loading1(self):
        self.controler.loading_text()

    def on_click(self):
        try:
            nlayers = int(self.nlayers_box.text())
            nfixed_material = int(self.nfixed_material_box.text())
            nmaterial_optim = int(self.nmaterial_optim_box.text())
            try: #modify for metasurfaces
                nfixed_metasurface = int(self.nfixed_metasurface_box.text())
                noptim_metasurface = int(self.noptim_metasurface_box.text())
            except:
                print(traceback.format_exc())
                nfixed_metasurface = 0
                noptim_metasurface = 0
            Lfiltering_index = self.LFfilter_choice.currentIndex()
            Hfiltering_index = self.HFfilter_choice.currentIndex()
            zeros_index = self.zeros_choice.currentIndex()
            dark_index = self.dark_choice.currentIndex()
            cutstart = float(self.start_box.text())
            cutend   = float(self.end_box.text())
            cutsharp = float(self.sharp_box.text())
            fitDelay = self.options_delay.currentIndex()
            delay_guess = 0
            delay_limit = 0
            delayfixed = False
            modesuper = 0
            fitLeftover = self.options_leftover.currentIndex()
            leftover_guess = np.zeros(2)
            leftover_limit = np.zeros(2)
            leftfixed = [False]*2
            slope = float(self.slope_box.text())
            intercept = float(self.intercept_box.text())
            try:
                if self.options_delay.currentIndex() == 1:
                    delay_guess = float(self.delay_guess_box.text())
                    delayfixed = self.fix_delay_box.isChecked()
                    if not delayfixed:
                        delay_limit = float(self.delay_limit_box.text())                    
                modesuper = self.options_super.currentIndex()
                if self.options_leftover.currentIndex() == 1:
                    leftover_guess[0] = float(self.leftovera_guess_box.text())
                    leftover_guess[1] = float(self.leftoverc_guess_box.text())
                    leftfixed[0] = self.fix_a_box.isChecked()
                    leftfixed[1] = self.fix_c_box.isChecked()
                    if not leftfixed[0]:
                        leftover_limit[0] = float(self.leftovera_limit_box.text())
                    if not leftfixed[1]:   
                        leftover_limit[1] = float(self.leftoverc_limit_box.text())
            except:
                print(traceback.format_exc())
                pass
#            if (nfixed_metasurface != 0) or (noptim_metasurface !=0):
#                self.controler.refreshAll("The software is not able to handle metasurfaces yet. You can add a transfer function, modify optimization and remove this warning.")
#                return(0)
            if nlayers<1:
                self.controler.refreshAll('Number of layers has to be more than 1')
                return(0)
            if (nfixed_material<0)|(nmaterial_optim<0)|(nfixed_metasurface<0)|(noptim_metasurface<0):
                self.controler.refreshAll('Numbers of materials should be positive')
                return(0)
            # if nmaterial_optim !=1:
            #     self.controler.refreshAll("The software is only able to handle one material to optimize.")
            #     return(0)
            #if noptim_metasurface >1: #err corrected
            #    self.controler.refreshAll("The software is only able to handle one metasurface to optimize.")
            #    return(0)
            try:
                self.controler.choices_ini(self.path_without_sample,self.path_with_sample,
                                               nlayers,nfixed_material, nmaterial_optim,
                                               nfixed_metasurface, noptim_metasurface,
                                               Lfiltering_index, Hfiltering_index, zeros_index, dark_index, cutstart, 
                                               cutend, cutsharp, slope, intercept, fitDelay, delay_guess, delay_limit, delayfixed, modesuper, fitLeftover, leftover_guess, leftover_limit, leftfixed)
                self.controler.refreshAll("Done.")
            except:
                print(traceback.format_exc())
                self.controler.error_message_path()
                self.controler.nlayers = 0
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.refreshAll("Invalid parameters, please enter real values only")

    def get_path_without_sample(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"Without sample", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"Without sample")
        try:
            self.path_without_sample=fileName
            name=os.path.basename(fileName)
            self.button_ask_path_without_sample.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path()

    def get_path_with_sample(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"With sample", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"With sample")
        try:
            self.path_with_sample=fileName
            Csts.FileName = fileName
            name=os.path.basename(fileName)
            self.button_ask_path_with_sample.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path()

    def refresh(self):
        pass


class TextBoxWidget(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.setReadOnly(True)
        self.append("Log")
        self.setMaximumWidth(440)
        
        references = [to_sup(1) + ' Metasurfaces are currently only implemented for 1 layer samples',
                      to_sup(2) + ' Time should be in ps.',
                      to_sup(3) + ' Sharpness: 100 is almost a step function, 0.1 is really smooth. See graphs in optimization tab.',
                      to_sup(4) + ' Calculate the delay between the two pulses maximum and set the end of the reference pulse to zero to avoid temporal aliasing.',
                      to_sup(5) + ' Use a linear function (or custom function) to remove the contribution of the dark noise after 200GHz.']
        for reference in references:
            self.append(reference)
            self.append('')

    def refresh(self):
        message = self.controler.message
        if message:
            self.append(message)
            
class layout_materials(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.setTitle("Materials")
        self.setMaximumHeight(290)

    def refresh(self):
        layout_materials=layout_materials_scroll(self,self.controler)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(layout_materials)

        self.main_layout=QVBoxLayout()
        self.main_layout.addWidget(scroll)
        if self.controler.nlayers:
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)
                
class layout_materials_scroll(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.nfixed_material = self.controler.nfixed_material
        self.nfixed_metasurface = self.controler.nfixed_metasurface
        if self.nfixed_material:
            self.materials = [Material() for i in range(self.nfixed_material)]
            self.hasMaterial = np.zeros(self.nfixed_material)
        else:
            self.materials = []
            self.hasMaterial = [1]
        self.interfaces = [TDS.Interface(name = 'dioptric interface')]
        if self.nfixed_metasurface:
            self.interfaces.append(TDS.Interface(isMetasurface = 1) for i in range(self.nfixed_metasurface))
            self.hasMetasurface = np.zeros(self.nfixed_metasurface)
        else:
            self.hasMetasurface = [1]
        
        text_box_width=120
        text_box_height=25
        
        ## Widgets and layouts
        # Error box
        self.log_box=log_layers(self,self.controler)
#        self.log_box.setMaximumWidth(300)
        # Name text
        self.label_name = QLabel('Name')
        self.label_name_2 = QLabel('Name')
        self.label_name_3 = QLabel('Name')
        self.label_name_4 = QLabel('Name')
        # File text
        self.label_file = QLabel('File')
        self.label_file_3 = QLabel('File')
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_data)
        self.submit_button.setMaximumHeight(text_box_height)
        
        self.text_boxes_fixed_material=  []
        self.buttons_file_material = []
        self.text_boxes_optim_material = []
        self.text_boxes_fixed_metasurface=  []
        self.buttons_file_metasurface = []
        self.text_boxes_optim_metasurface = []

        layouts_fixed_material = []
        layouts_fixed_metasurface = []

        if self.controler.nfixed_material:
            #boxes
            for i in range(self.controler.nfixed_material):
                self.text_boxes_fixed_material.append(QLineEdit())
                self.text_boxes_fixed_material[i].setMaximumWidth(text_box_width)
                self.text_boxes_fixed_material[i].setMaximumHeight(text_box_height)
                
                self.buttons_file_material.append(QPushButton("Browse"))
                self.buttons_file_material[i].setMaximumWidth(text_box_width)
                self.buttons_file_material[i].setMaximumHeight(text_box_height)
                self.index=i
                self.buttons_file_material[i].clicked.connect(self.search_file(i,1))
                
                layouts_fixed_material.append(QHBoxLayout())
                layouts_fixed_material[i].setAlignment(Qt.AlignTop)
        
        if self.controler.noptim_material:
            for i in range(self.controler.noptim_material):
                self.text_boxes_optim_material.append(QLineEdit())
                self.text_boxes_optim_material[i].setMaximumWidth(text_box_width)
                self.text_boxes_optim_material[i].setMaximumHeight(text_box_height)
        
        if self.controler.nfixed_metasurface:
            #boxes
            for i in range(self.controler.nfixed_metasurface):
                self.text_boxes_fixed_metasurface.append(QLineEdit())
                self.text_boxes_fixed_metasurface[i].setMaximumWidth(text_box_width)
                self.text_boxes_fixed_metasurface[i].setMaximumHeight(text_box_height)
                
                self.buttons_file_metasurface.append(QPushButton("Browse"))
                self.buttons_file_metasurface[i].setMaximumWidth(text_box_width)
                self.buttons_file_metasurface[i].setMaximumHeight(text_box_height)
                self.buttons_file_metasurface[i].clicked.connect(self.search_file(i,0))
                
                layouts_fixed_metasurface.append(QHBoxLayout())
                layouts_fixed_metasurface[i].setAlignment(Qt.AlignTop)
                
        if self.controler.noptim_metasurface:
            for i in range(self.controler.noptim_metasurface):
                self.text_boxes_optim_metasurface.append(QLineEdit())
                self.text_boxes_optim_metasurface[i].setMaximumWidth(text_box_width)
                self.text_boxes_optim_metasurface[i].setMaximumHeight(text_box_height)
                
        #default name, may be commented.
        if (self.controler.noptim_material == 1):
            self.text_boxes_optim_material[0].setText('sample')
                
        # Organisation Layout
        self.main_layout=QHBoxLayout()
        sub_layout_g=QGridLayout()
        sub_layout_g.setAlignment(Qt.AlignTop)
        
        if self.controler.nfixed_material:
            sub_layout_g1 = QGridLayout()
            sub_layout_g1.setAlignment(Qt.AlignTop)
            sub_layout_g1.addWidget(self.label_name,0,1)
            sub_layout_g1.addWidget(self.label_file,0,0)
            for i in range(self.controler.nfixed_material):
                sub_layout_g1.addWidget(self.text_boxes_fixed_material[i],i+1,1)
                sub_layout_g1.addWidget(self.buttons_file_material[i],i+1,0)
            # Group
            group_1 = QGroupBox()
            group_1.setTitle('Fixed materials')
            group_1.setLayout(sub_layout_g1)
            sub_layout_g.addWidget(group_1)
                
        if self.controler.noptim_material:
            sub_layout_v = QVBoxLayout()
            sub_layout_v.addWidget(self.label_name_2) 
            for i in range(self.controler.noptim_material):
                sub_layout_v.addWidget(self.text_boxes_optim_material[i])
            # Group
            group_2 = QGroupBox()
            group_2.setTitle('Materials to optimize')
            group_2.setLayout(sub_layout_v)
            sub_layout_g.addWidget(group_2)
            
        if self.controler.nfixed_metasurface:
            sub_layout_g3 = QGridLayout()
            sub_layout_g3.setAlignment(Qt.AlignTop)
            sub_layout_g3.addWidget(self.label_name_3,0,1)
            sub_layout_g3.addWidget(self.label_file_3,0,0)
            for i in range(self.controler.nfixed_metasurface):
                sub_layout_g3.addWidget(self.text_boxes_fixed_metasurface[i],i+1,1)
                sub_layout_g3.addWidget(self.buttons_file_metasurface[i],i+1,0)
            # Group
            group_3 = QGroupBox()
            group_3.setTitle('Fixed metasurfaces')
            group_3.setLayout(sub_layout_g3)
            sub_layout_g.addWidget(group_3)
                
        if self.controler.noptim_metasurface:
            sub_layout_v4 = QVBoxLayout()
            sub_layout_v4.addWidget(self.label_name_4) 
            for i in range(self.controler.noptim_metasurface):
                sub_layout_v4.addWidget(self.text_boxes_optim_metasurface[i])
            # Group
            group_4 = QGroupBox()
            group_4.setTitle('Metasurfaces to optimize')
            group_4.setLayout(sub_layout_v4)
            sub_layout_g.addWidget(group_4)
        
        if (self.controler.nlayers!=None):
            sub_layout_g.addWidget(self.submit_button)
            self.main_layout.addLayout(sub_layout_g)
            self.main_layout.addWidget(self.log_box)
            self.setLayout(self.main_layout)
        
    def refresh(self):
        pass
    
    def search_file(self,index,isMaterial):
        def search_file():
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog
            # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()","","All Files (*);;Python Files (*.py)", options=options)
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()","","All Files (*);;Python Files (*.py)")
            try:
                name=os.path.basename(fileName)
                if name =='':
                    self.log_box.append('Please enter a valid path')
                    return(0)
            except:
                print(traceback.format_exc())
                self.log_box.append('Please enter a valid path')
                return(0)
            try:
                if isMaterial:
                    self.materials[index] = Material(file = fileName)
                    self.buttons_file_material[index].setText(name)
                    self.text_boxes_fixed_material[index].setText(name.split('.')[0])
                    self.hasMaterial[index]=1
                else:
                    self.interfaces[index+1] = TDS.Interface(file = fileName)
                    self.buttons_file_metasurface[index].setText(name)
                    self.text_boxes_fixed_metasurface[index].setText(name.split('.')[0])
                    self.hasMetasurface[index]=1
            except:
                print(traceback.format_exc())
                self.log_box.append('There is an error with the file submitted')
        return search_file
    
    def submit_data(self):
        materialsNames = []
        metasurfacesNames = ['dioptric interface']

        for i in range(self.nfixed_material):
            if self.hasMaterial[i]==0:
                self.controler.refreshAll("Please enter all material files.")
                return(0)
        for i in range(self.nfixed_material):
            name = self.text_boxes_fixed_material[i].text()
            if name !='':
                materialsNames.append(name)
                # materials have been added to the list in search_path
            else:
                self.controler.refreshAll('Please give all the materials a name')
                return(0)
        for i in range(self.controler.noptim_material):
            name = self.text_boxes_optim_material[i].text()
            if name !='':
                materialsNames.append(name)
                self.materials.append(Material(name = name, fit_material = 1))
            else:
                self.controler.refreshAll('Please give all the materials a name')
                return(0)
        # Metasurfaces
        for i in range(self.nfixed_metasurface):
            if self.hasMetasurface[i]==0:
                self.controler.refreshAll("Please enter all metasurfaces files.")
                return(0)
            name = self.text_boxes_fixed_metasurface[i].text()
            if name !='':
                metasurfacesNames.append(name)
                # metasurfaces have been added to the list in search_path
            else:
                self.controler.refreshAll('Please give all the metasurfaces a name')
                return(0)
        for i in range(self.controler.noptim_metasurface):
            name = self.text_boxes_optim_metasurface[i].text()
            if name !='':
                metasurfacesNames.append(name)
                self.interfaces.append(TDS.Interface(name = name, isMetasurface = 1, fit_metasurface = 1))
            else:
                self.controler.refreshAll('Please give all the metasurfaces a name')
                return(0)
        
        if len(materialsNames)>0:
            materialsNames.append('air')
            self.materials.append(Material(name = 'air'))
            self.controler.materialList = self.materials
            self.controler.materialNames = materialsNames
            # Metasurfaces
            self.controler.distinctInterfaceList = self.interfaces
            self.controler.interfaceNames = metasurfacesNames
        else:
            self.controler.refreshAll('No materials given')
            return(0)
        self.controler.refreshAll('')
            
class log_layers(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.setReadOnly(True)
        self.append('Log')
    def refresh(self):
        pass

class layers(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient(self)
        self.setTitle("Layers")
        self.setMaximumHeight(400)

    def refresh(self):
        layers=layers_scroll(self,self.controler)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(layers)

        self.main_layout=QVBoxLayout()
        self.main_layout.addWidget(scroll)
        if self.controler.nlayers:
            if len(self.controler.materialList):
                try:
                    deleteLayout(self.layout())
                    self.layout().deleteLater()
                except AttributeError:
                    self.setLayout(self.main_layout)

class layers_scroll(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.controler.addClient(self)
        self.nlayers = self.controler.nlayers
        
        label_width=240
        text_box_width=120
        text_box_height=25
        
        ## Creation Widgets and Layouts
        # Error box
        self.log_box=log_layers(self,self.controler)
        self.log_box.setMaximumHeight(100)
        # labels
        self.label_number=QLabel('Layer')
        self.label_number.setMaximumWidth(label_width)
        
        self.label_thickness=QLabel('Thickness of the sample (m)')
        self.label_thickness.setMaximumWidth(label_width)
        
        self.label_uncertainty=QLabel('Uncertainty of the thickness (%)')
        self.label_uncertainty.setMaximumWidth(label_width)
        
        self.label_fit=QLabel('Fit thickness? '+to_sup(1))
        self.label_fit.setMaximumWidth(label_width)
        
        self.label_material=QLabel('Material')
        self.label_material.setMaximumWidth(label_width)
        
        self.label_interface=QLabel('Interface before layer')
        self.label_interface.setMaximumWidth(label_width)
        
        self.label_air=QLabel('air')
        self.label_air.setMaximumWidth(label_width)
        self.label_air2=QLabel('air')
        self.label_air.setMaximumWidth(label_width)
        
        self.label_1=QLabel('-')
        self.label_1.setMaximumWidth(label_width)
        self.label_2=QLabel('-')
        self.label_2.setMaximumWidth(label_width)
        self.label_3=QLabel('-')
        self.label_3.setMaximumWidth(label_width)

        self.label_FP=QLabel('Fabry Perot effect?')
        self.label_FP.setMaximumWidth(label_width)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_data)
        self.submit_button.pressed.connect(self.pressed_loading)
        self.submit_button.setMaximumWidth(text_box_width)
        self.submit_button.setMaximumHeight(text_box_height)
        
        self.labels = []
        self.thickness_boxes = []
        self.uncertainty_boxes = []
        self.fit_options = []
        self.materials_options = []
        self.interface_options = []
        self.FP_choice = []
        
        if len(self.controler.materialList):
            if (self.nlayers):
                # create widgets
                for i in range(self.nlayers):
                    self.labels.append(QLabel('{}'.format(i)))
                    self.labels[i].setMaximumWidth(text_box_width)
                    self.labels[i].setMaximumHeight(text_box_height)
                    
                    self.thickness_boxes.append(QLineEdit())
                    self.thickness_boxes[i].setMaximumWidth(text_box_width)
                    self.thickness_boxes[i].setMaximumHeight(text_box_height)
                    
                    self.uncertainty_boxes.append(QLineEdit())
                    self.uncertainty_boxes[i].setMaximumWidth(text_box_width)
                    self.uncertainty_boxes[i].setMaximumHeight(text_box_height)
                    
                    self.fit_options.append(QComboBox())
                    self.fit_options[i].addItems(['Yes', 'No'])
                    self.fit_options[i].setMaximumWidth(text_box_width)
                    self.fit_options[i].setMaximumHeight(text_box_height)
                    
                    self.materials_options.append(QComboBox())
                    self.materials_options[i].addItems(self.controler.materialNames)
                    self.materials_options[i].setMaximumWidth(text_box_width)
                    self.materials_options[i].setMaximumHeight(text_box_height)

                    self.FP_choice = QComboBox()
                    self.FP_choice.addItems(['Yes', 'No'])
                    self.FP_choice.setMaximumWidth(text_box_width)
                    self.FP_choice.setMaximumHeight(text_box_height)
                    
                for i in range(self.nlayers+1):
                    self.interface_options.append(QComboBox())
                    self.interface_options[i].addItems(self.controler.interfaceNames)
                    self.interface_options[i].setMaximumWidth(text_box_width)
                    self.interface_options[i].setMaximumHeight(text_box_height)
                
                if (self.nlayers == 3)&(self.controler.noptim_material ==1):
                    if (self.controler.nfixed_material == 1)|(self.controler.nfixed_material == 0):
                        self.materials_options[1].setCurrentIndex(1)
            
                self.main_layout = QVBoxLayout()
                sub_layout_g = QGridLayout()
                sub_layout_h = QHBoxLayout()
                sub_layout_h.addWidget(self.log_box)
                sub_layout_h.addWidget(self.submit_button)
                
                sub_layout_g.addWidget(self.label_number,     0,0)
                sub_layout_g.addWidget(self.label_thickness,  1,0)
                sub_layout_g.addWidget(self.label_uncertainty,2,0)
                sub_layout_g.addWidget(self.label_fit,        3,0)
                sub_layout_g.addWidget(self.label_FP,        4,0)
                sub_layout_g.addWidget(self.label_material,   5,0)
                sub_layout_g.addWidget(self.label_interface,  6,0)

                for i in range(self.nlayers):
                    sub_layout_g.addWidget(self.labels[i],           0,i+1)
                    sub_layout_g.addWidget(self.thickness_boxes[i],  1,i+1)
                    sub_layout_g.addWidget(self.uncertainty_boxes[i],2,i+1)
                    sub_layout_g.addWidget(self.fit_options[i],      3,i+1)
                    sub_layout_g.addWidget(self.FP_choice,      4,i+1)
                    sub_layout_g.addWidget(self.materials_options[i],5,i+1)
                for i in range(self.nlayers+1):
                    sub_layout_g.addWidget(self.interface_options[i],6,i+1)
                sub_layout_g.addWidget(self.label_air, 0,self.nlayers+1)
                sub_layout_g.addWidget(self.label_1,   1,self.nlayers+1)
                sub_layout_g.addWidget(self.label_2,   2,self.nlayers+1)
                sub_layout_g.addWidget(self.label_3,   3,self.nlayers+1)
                sub_layout_g.addWidget(self.label_air2,4,self.nlayers+1)
                
                
                self.main_layout.addLayout(sub_layout_g)
                self.main_layout.addLayout(sub_layout_h)
                self.setLayout(self.main_layout)
                self.log_box.append(to_sup(1) + ' Set the thickness of the layer as an optimization parameter.')
                
                
    def refresh(self):
        pass
    
    def pressed_loading(self):
        self.parent.parent.text_box.append("\n Processing...\n") # using refreshAll caused data to disappear
    
    def submit_data(self):
        layerlist = []
        position_optim_material = []
        position_optim_interface = []
        position_optim_thickness = []
        id_FP = []
        nbpi= 0
        # Get the layers
        id_FP = self.FP_choice.currentIndex()
        for i in range(self.controler.nlayers):
            try:
                thickness=float(self.thickness_boxes[i].text())
                uncertainty=float(self.uncertainty_boxes[i].text())
                if thickness<=0:
                    self.log_box.append("Warning: thickness should be positive")
                    return(0)
                if uncertainty<0 or uncertainty>100:
                    self.log_box.append("Warning: The uncertainty you entered is not between 0 an 100%.")
                    return(0)
            except:
                print(traceback.format_exc())
                self.log_box.append("Error: Please enter real numbers.")
                return(0)
            
            material = self.controler.materialList[self.materials_options[i].currentIndex()]
            if material.fit_material:
                position_optim_material.append(i)
            layer = Layer(thickness, material, uncertainty = uncertainty, 
                          fit_index = self.fit_options[i].currentIndex())
            layerlist.append(layer)
            if layer.fit_thickness:
                position_optim_thickness.append(i)
        #Get the interfaces
        for i in range(self.controler.nlayers+1):
            interface = self.controler.distinctInterfaceList[self.interface_options[i].currentIndex()]
            if interface.fit_metasurface:
                position_optim_interface.append(i)
            self.controler.interfaceList.append(interface)
        
        self.controler.param_ini(layerlist,position_optim_material,position_optim_thickness,position_optim_interface,id_FP, nbpi)
        self.controler.initialised=1
        if self.controler.nlayers ==1: #change param_ini to not need that? Get message from controler instead?
            self.controler.refreshAll("Data submitted")
        else:
            self.log_box.append("Data submitted") #data still written in the boxes.



###############################################################################
###############################################################################
######################  Model parameters tab   ################################
###############################################################################
###############################################################################

class Model_parameters_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.setMinimumSize(640, 480)

        # Creation widgets
        self.model_choices=model_choices(self,controler)
        self.references_widget=references_widget(self,controler)
        self.param_values=parameters_values(self,controler)

        # Creation Layouts
        main_layout=QHBoxLayout()
        sub_layout_v1=QVBoxLayout()
        sub_layout_v2=QVBoxLayout()

        # Organisation layouts
        sub_layout_v1.addWidget(self.model_choices)
        sub_layout_v1.addWidget(self.references_widget)

        sub_layout_v2.addWidget(self.param_values)

        main_layout.addLayout(sub_layout_v1)
        main_layout.addLayout(sub_layout_v2)
        self.setLayout(main_layout)

class model_choices(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
        self.setTitle("Model choices")
        self.setMaximumWidth(400)

    def refresh(self):
        nb_param=self.controler.noptim_material + self.controler.noptim_metasurface

        if nb_param == 0:
            deleteLayout(self.layout())
        if nb_param:
            mod_choices=model_choices_scroll(self,self.controler)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setWidget(mod_choices)
    
            self.main_layout=QVBoxLayout()
            self.main_layout.addWidget(scroll)
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)




class model_choices_scroll(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
        self.setMaximumWidth(400)

        # Creation widgets
        label_width=200
        action_widget_width=200
        corrective_width_factor=-16

#        # Photonic structre #remove
#        self.label_struct = QLabel("Photonic structure \u00b9")
#        self.label_struct.setMaximumWidth(label_width)
#        self.options_struct = QComboBox()
#        self.options_struct.addItems(['Transmission Fabry-Perot',
#                                      'Transmission Fabry-Perot \n with a resonator (TDCMT)'])
#        self.options_struct.setMaximumWidth(action_widget_width +
#                                            corrective_width_factor)
#        self.options_struct.setMaximumHeight(30)

        self.label_name = []
        
        self.label_model = []
        self.enter_model = []


       
        main_layout=QVBoxLayout()
        sub_layout_h0 = []
        sub_layout_h = []
        
        i=0 
        for material in self.controler.materialList:
            if material.fit_material == 1:
                # Name of the material
                self.label_name.append(QLabel("{} :".format(material.name)))
                self.label_name[i].setMaximumWidth(label_width)
                sub_layout_h0.append(QHBoxLayout())
                sub_layout_h0[i].addWidget(self.label_name[i],0)
                main_layout.addLayout(sub_layout_h0[i])

                self.label_model.append([])
                self.enter_model.append([])
                sub_layout_h.append([])
                for j in range(len(Model.materialModels)):
                    self.label_model[i].append(QLabel(Model.materialModels[j].label + to_sup(j+1)))
                    self.label_model[i][j].setMaximumWidth(label_width)
                    if Model.materialModels[j].isCumulative:
                        self.enter_model[i].append(QLineEdit())
                        self.enter_model[i][j].setMaximumWidth(action_widget_width + corrective_width_factor)
                        self.enter_model[i][j].setMaximumHeight(30)
                        self.enter_model[i][j].setText("0")
                    else:
                        self.enter_model[i].append(QComboBox())
                        self.enter_model[i][j].addItems(['No', 'Yes'])
                        self.enter_model[i][j].setMaximumWidth(action_widget_width)
                # Creation layouts
                    sub_layout_h[i].append(QHBoxLayout())

                # Organisation Layouts
                    sub_layout_h[i][j].addWidget(self.label_model[i][j],0)
                    sub_layout_h[i][j].addWidget(self.enter_model[i][j],0)

                    main_layout.addLayout(sub_layout_h[i][j])
                
                i+=1
        for interface in self.controler.interfaceList:
            if interface.fit_metasurface == 1:
                self.label_name.append(QLabel("{} :".format(interface.name)))
                self.label_name[i].setMaximumWidth(label_width)
                sub_layout_h0.append(QHBoxLayout())
                sub_layout_h0[i].addWidget(self.label_name[i],0)
                main_layout.addLayout(sub_layout_h0[i])
                
                self.label_model.append([])
                self.enter_model.append([])
                sub_layout_h.append([])
                for j in range(len(Model.interfaceModels)):
                    self.label_model[i].append(QLabel(Model.interfaceModels[j].label + to_sup(len(Model.materialModels)+j+1)))
                    self.label_model[i][j].setMaximumWidth(label_width)
                    self.enter_model[i].append(QLineEdit())
                    self.enter_model[i][j].setMaximumWidth(action_widget_width +corrective_width_factor)
                    self.enter_model[i][j].setMaximumHeight(30)
                    self.enter_model[i][j].setText("0")
                # Creation layouts
                    sub_layout_h[i].append(QHBoxLayout())

                # Organisation Layouts
                    sub_layout_h[i][j].addWidget(self.label_model[i][j],0)
                    sub_layout_h[i][j].addWidget(self.enter_model[i][j],0)

                    main_layout.addLayout(sub_layout_h[i][j])
                
                i+=1
                
        # OK button
        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(self.submit_model_param)
        
        main_layout.addWidget(self.button_submit)

        self.setLayout(main_layout)

    def submit_model_param(self):
        nbTerms = [] 

        for i in range(self.controler.noptim_material):
            nbTerms.append([])
            for j in range(len(Model.materialModels)):
                try:
                    if Model.materialModels[j].isCumulative:
                        nbTerms[i].append(int(self.enter_model[i][j].text()))
                    else:
                        nbTerms[i].append(self.enter_model[i][j].currentIndex())
                    if nbTerms[i][j]<0:
                        self.controler.invalid_n_model(Model.materialModels[j].invalidNumberMessage)
                        return(0)
                except:
                    print(traceback.format_exc())
                    self.controler.invalid_n_model(Model.materialModels[j].invalidNumberMessage)
                    return(0)
        for i in range(self.controler.noptim_material,self.controler.noptim_material+self.controler.noptim_metasurface):    #we add the numbers of interface models after the material models
            nbTerms.append([])
            for j in range(len(Model.interfaceModels)):
                try:
                    nbTerms[i].append(int(self.enter_model[i][j].text()))
                    if nbTerms[i][j]<0:
                        self.controler.invalid_n_model(Model.interfaceModels[j].invalidNumberMessage)
                        return(0)
                except:
                    print(traceback.format_exc())
                    self.controler.invalid_n_model(Model.interfaceModels[j].invalidNumberMessage)
                    return(0)
        try:
            if self.controler.initialised == 1:
                self.controler.reset_values()
                self.controler.parameters_values(nbTerms)
            else:
                self.controler.invalid_tun_opti_first()
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.invalid_param()
    def refresh(self):
        pass


class references_widget(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
        #self.setFixedHeight(200)
        self.setFixedWidth(400)
        # self.setReadOnly(True)
        self.append("References")
        references=[to_sup(j+1) + " " + Model.materialModels[j].explanation + "\n" for j in range(len(Model.materialModels))]
        references = references + [to_sup(len(Model.materialModels)+j+1) + " " + Model.interfaceModels[j].explanation + "\n" for j in range(len(Model.interfaceModels))]
        for i in references:
            self.append(i)

    def refresh(self):
        message = self.controler.message
        if message:
            self.append(message)




class parameters_values(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
        self.setTitle("Parameters Values")

    def refresh(self):
        nb_param=self.controler.nb_param

        if nb_param == 0:
            deleteLayout(self.layout())
        if nb_param:
            parameters_values=parameters_values_scroll(self,self.controler)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setWidget(parameters_values)
    
            self.main_layout=QVBoxLayout()
            self.main_layout.addWidget(scroll)
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)

class parameters_values_scroll(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)

        mydescription=self.controler.mydescription
        myvariables=self.controler.myvariables
        myunits=self.controler.myunits
        nb_param=self.controler.nb_param

        # Path of a file containing the values of the parameters (optional)
        self.path_param_values=None
        # Directory and name to save parameters in external file
        self.dir_save_param=None
        self.name_save_parameters=None

        label_width=1500
        text_box_width=100
        text_box_height=25


        # Creation Widgets and Layouts
        self.label_param=QLabel('Parameter')
        self.label_param.setMaximumWidth(label_width)

        self.label_value=QLabel('Value')
        self.label_value.setMaximumWidth(text_box_width)

        self.label_min=QLabel('Min')
        self.label_min.setMaximumWidth(text_box_width)

        self.label_max=QLabel('Max')
        self.label_max.setMaximumWidth(text_box_width)

        self.label_search_path=QLabel('Use file to enter parameters values (optional)')
        self.label_search_path.setMaximumWidth(label_width)

        self.labels=[]

        self.main_layout=QVBoxLayout()
        sub_layout_h=QHBoxLayout()
        sub_layout_h2=QHBoxLayout()
        sub_layout_h3=QHBoxLayout()
        sub_layout_h4=QHBoxLayout()
        sub_layout_h5=QHBoxLayout()
        sub_layout_h6=QHBoxLayout()
        sub_layout_h7=QHBoxLayout()
        layouts=[]

        self.text_boxes_value=[]
        self.text_boxes_min=[]
        self.text_boxes_max=[]

        self.search_path_button = QPushButton("browse")
        self.search_path_button.clicked.connect(self.search_path)
        self.search_path_button.setMaximumWidth(text_box_width)
        self.search_path_button.setMaximumHeight(text_box_height)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_values)
        self.submit_button.setMaximumWidth(text_box_width)
        self.submit_button.setMaximumHeight(text_box_height)
        
        # Widgets to enter the name of the file to save parameters
        self.label_name = QLabel('Name of the file')
        self.label_name.setMaximumWidth(label_width)
        self.enter_name = QLineEdit()
        self.enter_name.setMaximumWidth(text_box_width)
        
        # Widgets to enter file directory
        self.label_path = QLabel('Path of the file')
        self.label_path.setMaximumWidth(label_width)
        self.search_save_path_button = QPushButton("browse")
        self.search_save_path_button.clicked.connect(self.search_save_path)
        self.search_save_path_button.setMaximumWidth(text_box_width)
        self.search_save_path_button.setMaximumHeight(text_box_height)
        
        # Button to save parameters in the desired file
        self.save_button = QPushButton("Save parameters in file")
        self.save_button.clicked.connect(self.save_values)
        self.save_button.setMaximumHeight(text_box_height)

        self.log_box=log_param_values(self,self.controler)
        self.log_box.setMaximumWidth(400)
        self.log_box.setMaximumHeight(50)


        if nb_param:
            for i in range(nb_param):
                self.labels.append(QLabel('{0} ({1})'.format(mydescription[i],myunits[i])))
                self.labels[i].setMaximumWidth(label_width)
                self.labels[i].setAlignment(Qt.AlignTop)

                self.text_boxes_value.append(QLineEdit())
                self.text_boxes_value[i].setMaximumWidth(text_box_width)
                self.text_boxes_value[i].setMaximumHeight(text_box_height)

                self.text_boxes_min.append(QLineEdit())
                self.text_boxes_min[i].setMaximumWidth(text_box_width)
                self.text_boxes_min[i].setMaximumHeight(text_box_height)

                self.text_boxes_max.append(QLineEdit())
                self.text_boxes_max[i].setMaximumWidth(text_box_width)
                self.text_boxes_max[i].setMaximumHeight(text_box_height)

                layouts.append(QHBoxLayout())
                layouts[i].setAlignment(Qt.AlignTop)

        # Organisation layouts
        sub_layout_h.addWidget(self.label_param)
        sub_layout_h.addWidget(self.label_value)
        sub_layout_h.addWidget(self.label_min)
        sub_layout_h.addWidget(self.label_max)
        sub_layout_h.setAlignment(Qt.AlignTop)
        self.main_layout.addLayout(sub_layout_h)

        if nb_param:
            for i in range(nb_param):
                layouts[i].addWidget(self.labels[i])
                layouts[i].addWidget(self.text_boxes_value[i])
                layouts[i].addWidget(self.text_boxes_min[i])
                layouts[i].addWidget(self.text_boxes_max[i])
                self.main_layout.addLayout(layouts[i])

        sub_layout_h2.addWidget(self.label_search_path)
        sub_layout_h2.addWidget(self.search_path_button)

        sub_layout_h3.addWidget(self.log_box)
        sub_layout_h3.addWidget(self.submit_button)
        
        sub_layout_h5.addWidget(self.label_name)
        sub_layout_h5.addWidget(self.enter_name)
        
        sub_layout_h6.addWidget(self.label_path)
        sub_layout_h6.addWidget(self.search_save_path_button)
        
        sub_layout_h7.addWidget(self.save_button)
        
        self.main_layout.addLayout(sub_layout_h2)
        self.main_layout.addLayout(sub_layout_h3)
        self.main_layout.addLayout(sub_layout_h4)
        sub_layoutv = QVBoxLayout()
        sub_layoutv.addLayout(sub_layout_h5)
        sub_layoutv.addLayout(sub_layout_h6)
        sub_layoutv.addLayout(sub_layout_h7)
        files_group = QGroupBox()
        files_group.setTitle('Save in external file (optional)')
        files_group.setLayout(sub_layoutv)
        self.main_layout.addWidget(files_group)

        self.setLayout(self.main_layout)

    def refresh(self):
        pass

    def submit_values(self):
        nb_param=self.controler.nb_param
        mesparam = np.zeros([nb_param,3])
        error_interval = 0
        try:
            for i in range(nb_param):
                if float(self.text_boxes_value[i].text())<float(self.text_boxes_min[i].text()):
                    self.log_box.append("The value in line {0} has to be superior to the minimum !".format(i+1))
                    error_interval = 1
                if float(self.text_boxes_value[i].text())>float(self.text_boxes_max[i].text()):
                    self.log_box.append("The value in line {0} has to be inferior to the maximum !".format(i+1))
                    error_interval = 1
                mesparam[i,0]=float(self.text_boxes_value[i].text())
                mesparam[i,1]=float(self.text_boxes_min[i].text())
                mesparam[i,2]=float(self.text_boxes_max[i].text())
            if error_interval:
                self.log_box.append("Values not submitted ! There is a problem with the values and intervals of the parameters.")
                return(0)
        except:
            print(traceback.format_exc())
            self.log_box.append("Invalid values.")
            return(0)
        for i,layer in enumerate(self.controler.layerlist):                            #Here, all the variables are send to every fitted layers and interfaces, this could be improved
            if layer.material.fit_material == 1:
                self.controler.layerlist[i].material.change_param(mesparam[:,0],self.controler.myvariables,mesparam[:,1],mesparam[:,2])
        for i,interface in enumerate(self.controler.interfaceList):
            if interface.fit_metasurface == 1:
                self.controler.interfaceList[i].change_param(mesparam[:,0],self.controler.myvariables,mesparam[:,1],mesparam[:,2])
        self.controler.save_optimisation_param(mesparam)
        self.log_box.append("Values submitted")
        
    def save_values(self):
        nb_param=self.controler.nb_param
        mesparam = np.zeros([nb_param,3])
        error_interval = 0
        try:
            for i in range(nb_param):
                if float(self.text_boxes_value[i].text())<float(self.text_boxes_min[i].text()):
                    self.log_box.append("The value in line {0} has to be superior to the minimum !".format(i+1))
                    error_interval = 1
                if float(self.text_boxes_value[i].text())>float(self.text_boxes_max[i].text()):
                    self.log_box.append("The value in line {0} has to be inferior to the maximum !".format(i+1))
                    error_interval = 1
                mesparam[i,0]=float(self.text_boxes_value[i].text())
                mesparam[i,1]=float(self.text_boxes_min[i].text())
                mesparam[i,2]=float(self.text_boxes_max[i].text())
            if error_interval:
                self.log_box.append("Values not saved ! There is a problem with the values and intervals of the parameters.")
                return(0)
        except:
            print(traceback.format_exc())
            self.log_box.append("Invalid values.")
            return(0)
        name = self.enter_name.text()
        if name == '':
            self.log_box.append('Please enter a name')
            return(0)
        try:
            self.controler.save_optimisation_param_outside(mesparam,self.dir_save_param,name)
            self.log_box.append("Values saved")
        except:
            print(traceback.format_exc())
            self.log_box.append("Please enter a valid path")
            return(0)

    def search_path(self):
        if self.controler.initialised == 1:
            nb_param=self.controler.nb_param
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog
            # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
            try:
                self.path_param_values=fileName
                name=os.path.basename(fileName)
                self.search_path_button.setText(name)
                self.log_box.append("Values taken from "+name)
            except:
                print(traceback.format_exc())
                self.controler.error_message_path()
                return(0)

            try:
                self.controler.mesparam=np.loadtxt(self.path_param_values)
                mes_param=self.controler.mesparam
                if nb_param==len(mes_param[:,0]):
                    for i in range(nb_param):
                        self.text_boxes_value[i].setText('{0:.3E}'.format(mes_param[i,0]))
                        self.text_boxes_min[i].setText('{0:.3E}'.format(mes_param[i,1]))
                        self.text_boxes_max[i].setText('{0:.3E}'.format(mes_param[i,2]))
                else:
                    self.log_box.append("The file submitted does not have the same number of parameters as the model chosen.")
                    return(0)
            except:
                print(traceback.format_exc())
                self.log_box.append("There is a problem with the file submitted.")
                return(0)
        else:
            self.log_box.append("Please run the initialisation window first.")
    
    def search_save_path(self):
        #find path to save parameters
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.dir_save_param=str(DirectoryName)
            name=os.path.basename(str(DirectoryName))
            self.search_save_path_button.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path2()

class log_param_values(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
#        self.setFixedHeight(150)
        self.setReadOnly(True)
        self.append("Optimisation")
    def refresh(self):
        pass

###############################################################################
###############################################################################
#########################   Optimisation tab   ################################
###############################################################################
###############################################################################

class Optimization_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.controler=controler

        # Creation Widgets
        self.action_choice = QComboBox()
        self.action_choice.addItems(['Optimize','Create fictional sample'])
        self.action_choice.currentIndexChanged.connect(self.refresh)
        self.action_choice.setMaximumWidth(380)
        self.action_choice.setMaximumHeight(20)
        self.action_widget = Action_handler(self,controler)
        self.action_widget.refresh()
        self.error_bars = Error_bars_handler(self,controler)
        self.error_bars.refresh()
        self.phase_correction = phase_correction_handler(self,controler)
        self.phase_correction.refresh()
        self.optim_param = Optimization_parameters(self,controler)
        self.optim_param.refresh()
        self.log_widget = log_optimisation(self,controler)
        self.graph_widget = Graphs_optimisation(self,controler)
    
        # Preview button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.preview_button.pressed.connect(self.pressed_loading)
        self.preview_button.setMaximumWidth(380)
        self.preview_button.setMaximumHeight(20)
        
        sub_layout_v = QVBoxLayout()
        sub_layout_v.addWidget(self.action_choice,0)
        sub_layout_v.addWidget(self.preview_button,0)
        sub_layout_v.addWidget(self.phase_correction,0)
        sub_layout_v.addWidget(self.action_widget,0)
        sub_layout_v.addWidget(self.error_bars,0)
        sub_layout_v.addWidget(self.optim_param,0)
        sub_layout_v.addWidget(self.log_widget,0)
        sub_layout_v.setAlignment(Qt.AlignTop)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(sub_layout_v,0)
        main_layout.addWidget(self.graph_widget,1)
        self.setLayout(main_layout)
        
    def preview(self):
        self.controler.preview()
    def pressed_loading(self):
        self.controler.loading_text3()
        
    def refresh(self):# /!\ Optimization_tab is not a client of the controler, 
    #this is not called by the controler, only when action_choice is changed.
        deleteLayout(self.action_widget.layout())
        self.action_widget.refresh()
        self.optim_param.refresh()
        self.controler.refreshAll3('')

class Action_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setMaximumWidth(440)

    def refresh(self):
        if self.parent.action_choice.currentIndex() == 0:
            self.action_widget = Optimization_choices(self,self.controler)
        else:
            self.action_widget = FictionalSample_choices(self,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except AttributeError:
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.action_widget)
            self.setLayout(main_layout)

class Optimization_choices(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler=controler
        self.parent = parent
        self.emptycellfile = None
        # Corrective factors
        label_width=200
        action_widget_width=150
        corrective_width_factor=-12
        
        # Widget to see output directory
        self.label_outputdir = QLabel('Output directory: ')
        self.label_outputdir.setMaximumWidth(label_width)
        self.button_outputdir = QPushButton('browse', self)
        self.button_outputdir.clicked.connect(self.get_outputdir)
        self.button_outputdir.setMaximumWidth(action_widget_width + corrective_width_factor)
        self.button_outputdir.setMaximumHeight(30)


        # Widget to name time domain output
        self.label_time_domain = QLabel('Time domain output filename: ')
        self.label_time_domain.setMaximumWidth(label_width)
        self.enter_time_domain = QLineEdit()
        self.enter_time_domain.setMaximumWidth(action_widget_width + corrective_width_factor)
        self.enter_time_domain.setMaximumHeight(30)


        # Widget to name frequency domain output
        self.label_frequency_domain = QLabel('Frequency domain output filename: ')
        self.label_frequency_domain.setMaximumWidth(label_width)
        self.enter_frequency_domain = QLineEdit()
        self.enter_frequency_domain.setMaximumWidth(action_widget_width + corrective_width_factor)
        self.enter_frequency_domain.setMaximumHeight(30)


        # Widget to name optimization output
        self.label_out_opt = QLabel('Output optimization: ')
        self.label_out_opt.setMaximumWidth(label_width)
        self.enter_out_opt = QLineEdit()
        self.enter_out_opt.setMaximumWidth(action_widget_width + corrective_width_factor)
        self.enter_out_opt.setMaximumHeight(30)
        
        
        # Algorithm choice
        self.label_algo = QLabel("Algorithm")
        self.label_algo.setMaximumWidth(label_width)
        self.options_algo = QComboBox()
        self.options_algo.addItems(['NumPy optimize swarm particle',
                                    'ALPSO without parallelization',
                                    'ALPSO with parallelization',
                                    'SLSQP (pyOpt)',
                                    'SLSQP (pyOpt with parallelization)',
                                    'L-BFGS-B',
                                    'SLSQP (scipy)',
                                    'Dual annealing'])
        self.options_algo.setMaximumWidth(action_widget_width+40)
        self.options_algo.currentIndexChanged.connect(self.refresh_param)
        
        
        # Target permittivity
        self.label_eps = QLabel("Compute the permittivity: ")
        self.reference_label = QLabel('Empty cell (if 3 layers)')
        self.reference_button = QPushButton("browse")
        self.reference_button.setMaximumHeight(20)
        self.reference_button.setMaximumWidth(action_widget_width)
        self.reference_button.clicked.connect(self.search_file)
        
        self.eps_init_button = QPushButton("with initial length")
        self.eps_init_button.setMaximumHeight(20)
        self.eps_init_button.clicked.connect(self.compute_eps_ini)
        self.eps_init_button.pressed.connect(self.pressed_loading)
        self.eps_opti_button = QPushButton("with the last optimum length")
        self.eps_opti_button.setMaximumHeight(20)
        self.eps_opti_button.clicked.connect(self.compute_eps_opti)
        self.eps_opti_button.pressed.connect(self.pressed_loading)
        
        # Error choice
        self.label_error = QLabel("Error function weighting "+to_sup(1))
        self.label_error.setMaximumWidth(label_width)
        self.options_error = QComboBox()
        self.options_error.addItems(['Constant','Custom weight', 'Custom noise', 'Noise matrix'])
        self.options_error.setMaximumWidth(action_widget_width-12)
        self.options_error.currentIndexChanged.connect(self.refresh_param)
        
        # # save error bars buttons #NOUREDDIN
        # self.save_error_bars_button = QPushButton("Save Error bars") 
        # self.save_error_bars_button.setMaximumHeight(20)
        # self.save_error_bars_button.setMaximumWidth(action_widget_width)
        # self.save_error_bars_button.clicked.connect(lambda: self.get_std_files_dialog.exec_()) #execute and show the dialog
        
        # # save error bars for models #NOUREDDIN
        # self.model_errors_checkbox = QCheckBox("Model errors")
        
        # self.get_std_files_dialog = QDialog()
        # self.get_std_files_dialog.setWindowTitle(f"Load Normalization files")
        # self.get_std_files_dialog.resize(750,200)
        
        
        # std_ref_group = QGroupBox(f"Reference std file")
        # std_ref_group.setMaximumSize(800,100)
        # std_ref_group.setMinimumSize(800,100)
        
        # std_sample_group = QGroupBox(f"Sample std file")
        # std_sample_group.setMaximumSize(800,100)
        # std_sample_group.setMinimumSize(800,100)
        
        # saving_dir_group = QGroupBox(f"Saving directory")
        # saving_dir_group.setMaximumSize(800,100)
        # saving_dir_group.setMinimumSize(800,100)
        
        # std_ref_layout = QHBoxLayout()
        # std_sample_layout = QHBoxLayout()
        # saving_dir_layout = QHBoxLayout()
        # dialog_main_layout = QVBoxLayout()
        
        
        # self.dialog_ref_textedit =  QTextEdit()
        # self.dialog_ref_textedit.setMaximumHeight(30)
        # self.dialog_ref_textedit.setMaximumWidth(750)
        # self.dialog_ref_textedit.setEnabled(False)
        
        # self.dialog_sample_textedit =  QTextEdit()
        # self.dialog_sample_textedit.setMaximumHeight(30)
        # self.dialog_sample_textedit.setMaximumWidth(750)
        # self.dialog_sample_textedit.setEnabled(False)
        
        # self.saving_dir_textedit =  QTextEdit()
        # self.saving_dir_textedit.setMaximumHeight(30)
        # self.saving_dir_textedit.setMaximumWidth(750)
        # self.saving_dir_textedit.setEnabled(False)
        
        # self.browse_ref = QPushButton(f"browse")
        # self.browse_ref.setMaximumHeight(30)
        # self.browse_ref.setMaximumWidth(75)
        
        # self.browse_sample = QPushButton(f"browse")
        # self.browse_sample.setMaximumHeight(30)
        # self.browse_sample.setMaximumWidth(75)
        
        # self.save_dir_button = QPushButton(f"browse")
        # self.save_dir_button.setMaximumHeight(30)
        # self.save_dir_button.setMaximumWidth(75)
        
        
        # # self.ref_label = QLabel(f"Reference std file")
        # # self.sample_label = QLabel(f"Sample std file")
        
        # self.ok_button = QPushButton(f"OK")
        # self.ok_button.setMaximumHeight(30)
        # self.ok_button.setMaximumWidth(75)
        
        # std_ref_layout.addWidget(self.dialog_ref_textedit)
        # std_ref_layout.addWidget(self.browse_ref)
        
        # std_sample_layout.addWidget(self.dialog_sample_textedit)
        # std_sample_layout.addWidget(self.browse_sample)
        
        # saving_dir_layout.addWidget(self.saving_dir_textedit)
        # saving_dir_layout.addWidget(self.save_dir_button)
        
        
        # std_ref_group.setLayout(std_ref_layout)
        # std_sample_group.setLayout(std_sample_layout)
        # saving_dir_group.setLayout(saving_dir_layout)
        
        
        # dialog_main_layout.addWidget(std_ref_group)
        # dialog_main_layout.addWidget(std_sample_group)
        # dialog_main_layout.addWidget(saving_dir_group)
        # dialog_main_layout.addWidget(self.ok_button)
        
        # self.get_std_files_dialog.setLayout(dialog_main_layout)
        
        

        # Creation layouts
        sub_layout_h=QHBoxLayout()
        sub_layout_h_2=QHBoxLayout()
        sub_layout_h_3=QHBoxLayout()
        sub_layout_h_4=QHBoxLayout()
        sub_layout_h_5=QHBoxLayout()
        sub_layout_h_6=QHBoxLayout()
        sub_layout_h_7=QHBoxLayout()
        sub_layout_h_8=QHBoxLayout()
        # sub_layout_h_9=QHBoxLayout() # save error bars buttons #NOUREDDIN
        # sub_layout_h_10=QHBoxLayout() # save error bars for models #NOUREDDIN
        main_layout=QVBoxLayout()
        

        # Organisation layouts
        
        # Permittivity
        sub_layout_h.addWidget(self.reference_label)
        sub_layout_h.addWidget(self.reference_button)
        sub_layout_h_2.addWidget(self.eps_init_button)
        sub_layout_h_2.addWidget(self.eps_opti_button)
        
        sub_layout_h_3.addWidget(self.label_outputdir)
        sub_layout_h_3.addWidget(self.button_outputdir)
        sub_layout_h_4.addWidget(self.label_time_domain)
        sub_layout_h_4.addWidget(self.enter_time_domain)
        sub_layout_h_5.addWidget(self.label_frequency_domain)
        sub_layout_h_5.addWidget(self.enter_frequency_domain)
        sub_layout_h_6.addWidget(self.label_out_opt)
        sub_layout_h_6.addWidget(self.enter_out_opt)
        
        # Organisation layouts for optimisation
        sub_layout_h_7.addWidget(self.label_algo)
        sub_layout_h_7.addWidget(self.options_algo)
        
        sub_layout_h_8.addWidget(self.label_error)
        sub_layout_h_8.addWidget(self.options_error)
        
        # sub_layout_h_9.addWidget(self.save_error_bars_button) # save error bars buttons #NOUREDDIN

        # Vertical layout        
        main_layout.addWidget(self.label_eps)
        #if (self.controler.nlayers == 3)&(self.controler.nfixed_material == 1):
        main_layout.addLayout(sub_layout_h)
        main_layout.addLayout(sub_layout_h_2)
        main_layout.addLayout(sub_layout_h_3)
        main_layout.addLayout(sub_layout_h_4)
        main_layout.addLayout(sub_layout_h_5)
        main_layout.addLayout(sub_layout_h_6)
        # optimize
        main_layout.addLayout(sub_layout_h_7)
        main_layout.addLayout(sub_layout_h_8)
        
        # # main_layout.addLayout(sub_layout_h_9) # save error bars buttons #NOUREDDIN
        
        self.setLayout(main_layout)

    def get_outputdir(self):
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.outputdir=str(DirectoryName)
            name=os.path.basename(str(DirectoryName))
            self.button_outputdir.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path3()

    def pressed_loading(self):
        self.controler.loading_text3()

    
    def search_file(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        try:
            name=os.path.basename(fileName)
            if name =='':
                self.controler.refreshAll3('Please enter a valid path')
                return(0)
            self.reference_button.setText(name)
            self.emptycellfile = fileName
        except:
            print(traceback.format_exc())
            self.controler.refreshAll3('Please enter a valid path')
            return(0)
    
    def compute_eps_ini(self, nbpi):
        self.nbpi=0
        self.controler.compute_eps_init(self.emptycellfile, self.nbpi)
    
    def compute_eps_opti(self):
        self.controler.compute_eps_opti(self.emptycellfile)
    
    def refresh_param(self):
        self.parent.parent.optim_param.refresh()
        self.controler.errorIndex = self.options_error.currentIndex() # for graph

class Error_bars_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setMaximumWidth(350)

    def refresh(self):
        self.error_widget = Error_bars(self,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except AttributeError:
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.error_widget)
            self.setLayout(main_layout)

class Error_bars(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setTitle(f"Error bars")
    
    # def refresh(self):
        self.ref_added = False
        self.sample_added = False
        
        
        
        label_width = 75
        textedit_width = 260
        height = 20
        button_width = 25
        
        sub_layout_h_1 = QHBoxLayout()
        sub_layout_h_2 = QHBoxLayout()
        sub_layout_h_3 = QHBoxLayout()
        sub_layout_h_4 = QHBoxLayout()
        sub_layout_h_5 = QHBoxLayout()
        main_layout = QVBoxLayout()
        
        self.index_error_check = QCheckBox(f"Index error")
        sub_layout_h_1.addWidget(self.index_error_check)
        
        self.model_error_check = QCheckBox(f"Model error")
        sub_layout_h_1.addWidget(self.model_error_check)
        # sub_layout_h_2.addWidget(self.model_error_check)
        
        self.stdref_label = QLabel(f"std ref")
        self.stdref_label.setMaximumSize(label_width,height)
        
        self.stdref_textedit = QLineEdit()
        self.stdref_textedit.setMaximumSize(textedit_width,height)
        self.stdref_textedit.setEnabled(False)
        
        self.stdref_browse = QPushButton("...") 
        self.stdref_browse.setMaximumSize(button_width,height)
        self.stdref_browse.setEnabled(False)
        
        
        sub_layout_h_3.addWidget(self.stdref_label)
        sub_layout_h_3.addWidget(self.stdref_textedit)
        sub_layout_h_3.addWidget(self.stdref_browse)


        self.stdsample_label = QLabel(f"std sample")
        self.stdsample_label.setMaximumSize(label_width,height)
        
        self.stdsample_textedit = QLineEdit()
        self.stdsample_textedit.setMaximumSize(textedit_width,height)
        self.stdsample_textedit.setEnabled(False)
        
        self.stdsample_browse = QPushButton("...") 
        self.stdsample_browse.setMaximumSize(button_width,height)
        self.stdsample_browse.setEnabled(False)
        
        
        sub_layout_h_4.addWidget(self.stdsample_label)
        sub_layout_h_4.addWidget(self.stdsample_textedit)
        sub_layout_h_4.addWidget(self.stdsample_browse)
        
        self.submit_choices_button = QPushButton(f"Submit")
        self.submit_choices_button.setMaximumSize(75,height)
        self.submit_choices_button.setEnabled(False)
        
        # sub_layout_h_5.addWidget(self.submit_choices_button)
        sub_layout_h_1.addWidget(self.submit_choices_button)
        
        
        main_layout.addLayout(sub_layout_h_1)
        main_layout.addLayout(sub_layout_h_2)
        main_layout.addLayout(sub_layout_h_3)
        main_layout.addLayout(sub_layout_h_4)
        main_layout.addLayout(sub_layout_h_5)
        
        
        self.setLayout(main_layout)
        # self.setMaximumHeight(350)
        
        self.index_error_check.stateChanged.connect(self.err_change_choice)
        self.model_error_check.stateChanged.connect(self.err_change_choice)
        self.stdref_browse.clicked.connect(self.browse_std_ref_file)
        self.stdsample_browse.clicked.connect(self.browse_std_sample_file)
        self.submit_choices_button.clicked.connect(self.submit_err_params)
        
        self.err_change_choice()
        
    def submit_err_params(self):
        temp_file_dir = path_(ROOT_DIR).joinpath(f"temp")
        if not temp_file_dir.is_dir():
            path_(temp_file_dir).mkdir()
        
        temp_file = temp_file_dir.joinpath(f"temp_err_bool.bin")
        index_errors_bool = self.index_error_check.isChecked()
        model_errors_bool = self.model_error_check.isChecked()
        bools = [index_errors_bool, model_errors_bool]
        with open(temp_file, 'wb') as f:
            # pickle.dump(index_errors_bool,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(bools,f,pickle.HIGHEST_PROTOCOL)
        
        
        err_params = temp_file_dir.joinpath(f"temp_err_files.bin")
        filepaths = [self.stdref_textedit.text(), self.stdsample_textedit.text()]
        with open(err_params, "wb") as file:
            pickle.dump(filepaths,file,pickle.HIGHEST_PROTOCOL)
        
    
    def browse_std_ref_file(self):
        ref_name, _ = QFileDialog.getOpenFileName(parent=self, caption= f"Select the correct@tds std_freq file for the reference", directory=f"{ROOT_DIR}",filter="text file (*.txt)")
        self.stdref_textedit.setText(f"{ref_name}")
        self.stdref_textedit.setEnabled(True)
        self.ref_added = True
        if self.ref_added and  self.sample_added:
            self.submit_choices_button.setEnabled(True)
    
    def browse_std_sample_file(self):
        sample_name, _ = QFileDialog.getOpenFileName(parent=self, caption= f"Select the correct@tds std_freq file for the sample", directory=f"{ROOT_DIR}",filter="text file (*.txt)")
        self.stdsample_textedit.setText(f"{sample_name}")
        self.stdsample_textedit.setEnabled(True)
        self.sample_added = True
        if self.sample_added and self.ref_added :
            self.submit_choices_button.setEnabled(True)
            
    
    def err_change_choice(self):

        index_errors_bool = self.index_error_check.isChecked()
        model_errors_bool = self.model_error_check.isChecked()
        
        if index_errors_bool:
            self.stdref_browse.setEnabled(True)
            self.stdsample_browse.setEnabled(True)
        else:
            self.stdref_browse.setEnabled(False)
            self.stdsample_browse.setEnabled(False)
        
        if model_errors_bool:
            self.submit_choices_button.setEnabled(True)
        elif not model_errors_bool and not self.ref_added and not self.sample_added:
            self.submit_choices_button.setEnabled(False)
            
    
    # def model_err_change_choice(self):
        # 
        # index_errors_bool = self.index_error_check.isChecked()
        # temp_mod_file = path_(ROOT_DIR).joinpath(f"temp").joinpath(f"temp_model_err.bin")
        # model_errors_bool = self.model_error_check.isChecked()
        # 
        # with open(temp_mod_file, 'wb') as f:
            # pickle.dump(model_errors_bool,f,pickle.HIGHEST_PROTOCOL)




class phase_correction_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setMaximumWidth(380)

    def refresh(self):
        self.phase_widget = phase_correction(self,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except AttributeError:
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.phase_widget)
            self.setLayout(main_layout)

class phase_correction(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.emptycellfile = None
        self.setTitle(f"Phase correction")

        self.label_phase = QLabel("Number of pi correction")
        self.enter_phase = QLineEdit()
        self.enter_phase.setMaximumWidth(50)
        self.enter_phase.setMaximumHeight(30)
        self.submit_phase = QPushButton("Submit")
        self.submit_phase.clicked.connect(self.compute_eps_phase_correction)
        self.submit_phase.pressed.connect(self.pressed_loading)
        self.submit_phase.setMaximumWidth(50)
        self.submit_phase.setMaximumHeight(20)

        # Creation layouts
        sub_layout_h=QHBoxLayout()
        main_layout=QVBoxLayout()
        sub_layout_h.addWidget(self.label_phase)
        sub_layout_h.addWidget(self.enter_phase)
        sub_layout_h.addWidget(self.submit_phase)
        main_layout.addLayout(sub_layout_h)

        self.setLayout(main_layout)

    def compute_eps_phase_correction(self):
        self.nbpi=float(self.enter_phase.text())
        nbpi=self.nbpi
        f=open(os.path.join("temp",'temp_file_phase.bin'),'wb')
        pickle.dump(nbpi,f,pickle.HIGHEST_PROTOCOL)
        f.close()
        self.controler.compute_eps_phase_corraction(self.emptycellfile)

    def pressed_loading(self):
        self.controler.loading_text3()
    
        
class Error_bars_handler(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setMaximumWidth(350)

    def refresh(self):
        self.error_widget = Error_bars(self,self.controler)
        try:
            deleteLayout(self.layout())
            self.layout().deleteLater()
        except AttributeError:
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.error_widget)
            self.setLayout(main_layout)

class Error_bars(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setTitle(f"Error bars")
    
    # def refresh(self):
        self.ref_added = False
        self.sample_added = False
        
        
        
        label_width = 75
        textedit_width = 260
        height = 20
        button_width = 25
        
        sub_layout_h_1 = QHBoxLayout()
        sub_layout_h_2 = QHBoxLayout()
        sub_layout_h_3 = QHBoxLayout()
        sub_layout_h_4 = QHBoxLayout()
        sub_layout_h_5 = QHBoxLayout()
        main_layout = QVBoxLayout()
        
        self.index_error_check = QCheckBox(f"Index error")
        sub_layout_h_1.addWidget(self.index_error_check)
        
        self.model_error_check = QCheckBox(f"Model error")
        sub_layout_h_1.addWidget(self.model_error_check)
        # sub_layout_h_2.addWidget(self.model_error_check)
        
        self.stdref_label = QLabel(f"std ref")
        self.stdref_label.setMaximumSize(label_width,height)
        
        self.stdref_textedit = QLineEdit()
        self.stdref_textedit.setMaximumSize(textedit_width,height)
        self.stdref_textedit.setEnabled(False)
        
        self.stdref_browse = QPushButton("...") 
        self.stdref_browse.setMaximumSize(button_width,height)
        self.stdref_browse.setEnabled(False)
        
        
        sub_layout_h_3.addWidget(self.stdref_label)
        sub_layout_h_3.addWidget(self.stdref_textedit)
        sub_layout_h_3.addWidget(self.stdref_browse)


        self.stdsample_label = QLabel(f"std sample")
        self.stdsample_label.setMaximumSize(label_width,height)
        
        self.stdsample_textedit = QLineEdit()
        self.stdsample_textedit.setMaximumSize(textedit_width,height)
        self.stdsample_textedit.setEnabled(False)
        
        self.stdsample_browse = QPushButton("...") 
        self.stdsample_browse.setMaximumSize(button_width,height)
        self.stdsample_browse.setEnabled(False)
        
        
        sub_layout_h_4.addWidget(self.stdsample_label)
        sub_layout_h_4.addWidget(self.stdsample_textedit)
        sub_layout_h_4.addWidget(self.stdsample_browse)
        
        self.submit_choices_button = QPushButton(f"Submit")
        self.submit_choices_button.setMaximumSize(75,height)
        self.submit_choices_button.setEnabled(False)
        
        # sub_layout_h_5.addWidget(self.submit_choices_button)
        sub_layout_h_1.addWidget(self.submit_choices_button)
        
        
        main_layout.addLayout(sub_layout_h_1)
        main_layout.addLayout(sub_layout_h_2)
        main_layout.addLayout(sub_layout_h_3)
        main_layout.addLayout(sub_layout_h_4)
        main_layout.addLayout(sub_layout_h_5)
        
        
        self.setLayout(main_layout)
        # self.setMaximumHeight(350)
        
        self.index_error_check.stateChanged.connect(self.err_change_choice)
        self.model_error_check.stateChanged.connect(self.err_change_choice)
        self.stdref_browse.clicked.connect(self.browse_std_ref_file)
        self.stdsample_browse.clicked.connect(self.browse_std_sample_file)
        self.submit_choices_button.clicked.connect(self.submit_err_params)
        
        self.err_change_choice()
        
    def submit_err_params(self):
        temp_file_dir = path_(ROOT_DIR).joinpath(f"temp")
        if not temp_file_dir.is_dir():
            path_(temp_file_dir).mkdir()
        
        temp_file = temp_file_dir.joinpath(f"temp_err_bool.bin")
        index_errors_bool = self.index_error_check.isChecked()
        model_errors_bool = self.model_error_check.isChecked()
        bools = [index_errors_bool, model_errors_bool]
        with open(temp_file, 'wb') as f:
            # pickle.dump(index_errors_bool,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(bools,f,pickle.HIGHEST_PROTOCOL)
        
        
        err_params = temp_file_dir.joinpath(f"temp_err_files.bin")
        filepaths = [self.stdref_textedit.text(), self.stdsample_textedit.text()]
        with open(err_params, "wb") as file:
            pickle.dump(filepaths,file,pickle.HIGHEST_PROTOCOL)
        
    
    def browse_std_ref_file(self):
        ref_name, _ = QFileDialog.getOpenFileName(parent=self, caption= f"Select the correct@tds std_freq file for the reference", directory=f"{ROOT_DIR}",filter="text file (*.txt)")
        self.stdref_textedit.setText(f"{ref_name}")
        self.stdref_textedit.setEnabled(True)
        self.ref_added = True
        if self.ref_added and  self.sample_added:
            self.submit_choices_button.setEnabled(True)
    
    def browse_std_sample_file(self):
        sample_name, _ = QFileDialog.getOpenFileName(parent=self, caption= f"Select the correct@tds std_freq file for the sample", directory=f"{ROOT_DIR}",filter="text file (*.txt)")
        self.stdsample_textedit.setText(f"{sample_name}")
        self.stdsample_textedit.setEnabled(True)
        self.sample_added = True
        if self.sample_added and self.ref_added :
            self.submit_choices_button.setEnabled(True)
            
    
    def err_change_choice(self):

        index_errors_bool = self.index_error_check.isChecked()
        model_errors_bool = self.model_error_check.isChecked()
        
        if index_errors_bool:
            self.stdref_browse.setEnabled(True)
            self.stdsample_browse.setEnabled(True)
        else:
            self.stdref_browse.setEnabled(False)
            self.stdsample_browse.setEnabled(False)
        
        if model_errors_bool:
            self.submit_choices_button.setEnabled(True)
        elif not model_errors_bool and not self.ref_added and not self.sample_added:
            self.submit_choices_button.setEnabled(False)
            
    
    # def model_err_change_choice(self):
        # 
        # index_errors_bool = self.index_error_check.isChecked()
        # temp_mod_file = path_(ROOT_DIR).joinpath(f"temp").joinpath(f"temp_model_err.bin")
        # model_errors_bool = self.model_error_check.isChecked()
        # 
        # with open(temp_mod_file, 'wb') as f:
            # pickle.dump(model_errors_bool,f,pickle.HIGHEST_PROTOCOL)




class Optimization_parameters(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.parent = parent
        self.controler = controler
        self.setTitle("Optimization parameters")

    def refresh(self):
        self.optim_index = self.parent.action_choice.currentIndex()
        if self.optim_index == 0:
            self.errorFile = None
            self.choices = self.parent.action_widget.action_widget
            self.algo_index = self.choices.options_algo.currentIndex()
            self.errorweight_index = self.choices.options_error.currentIndex()
            # Corrective factors
            action_widget_width=150
            corrective_width_factor=-12
            # SwarmSize
            self.label_swarmsize = QLabel("Swarmsize")
            self.enter_swarmsize = QLineEdit()
            self.enter_swarmsize.setMaximumWidth(action_widget_width +corrective_width_factor)
            self.enter_swarmsize.setMaximumHeight(30)
    
            # Number of iterations
            self.label_niter = QLabel("Number of iterations")
            self.enter_niter = QLineEdit()
            self.enter_niter.setMaximumWidth(action_widget_width +corrective_width_factor)
            self.enter_niter.setMaximumHeight(30)
    
            # Wiget to see how many process are going to be used for the omptimization
            self.label_nb_proc = QLabel("How many process do you want to use?")
            self.enter_nb_proc = QLineEdit()
            self.enter_nb_proc.setText('1')
            self.enter_nb_proc.setMaximumWidth(50)
            self.enter_nb_proc.setMaximumHeight(25)
            
            # Files needed for error function
            if self.errorweight_index == 1:
                self.label_customweight = QLabel('Weight')
                self.button_errorFile = QPushButton('browse', self)
                self.button_errorFile.clicked.connect(self.get_weight)
                self.button_errorFile.setMaximumWidth(action_widget_width + corrective_width_factor)
                self.button_errorFile.setMaximumHeight(30)
                
            elif self.errorweight_index == 2:
                self.label_customweight = QLabel('Noise')
                self.button_errorFile = QPushButton('browse', self)
                self.button_errorFile.clicked.connect(self.get_weight)
                self.button_errorFile.setMaximumWidth(action_widget_width + corrective_width_factor)
                self.button_errorFile.setMaximumHeight(30)
                
            elif self.errorweight_index == 3:
                self.label_customweight = QLabel('Noise matrix')
                self.button_errorFile = QPushButton('browse', self)
                self.button_errorFile.clicked.connect(self.get_weight)
                self.button_errorFile.setMaximumWidth(action_widget_width + corrective_width_factor)
                self.button_errorFile.setMaximumHeight(30)
            
            # Button to launch optimization
            self.begin_button = QPushButton("Begin")
            self.begin_button.clicked.connect(self.begin_optimization)
            self.begin_button.pressed.connect(self.pressed_loading)
            #self.begin_button.setMaximumWidth(50)
            self.begin_button.setMaximumHeight(20)
            
            #button to test error guess
#            self.testguess_button = QPushButton("Test Guess")
#            self.testguess_button.clicked.connect(self.optim.guesstest())
#            self.testguess_button.pressed.connect(self.pressed_loading)
#            self.testguess_button.setMaximumHeight(20)
            
            sub_layout_h1=QHBoxLayout()
            sub_layout_h2=QHBoxLayout()
            sub_layout_h3=QHBoxLayout()
            sub_layout_h4=QHBoxLayout()
            
            sub_layout_h1.addWidget(self.label_nb_proc,0)
            sub_layout_h1.addWidget(self.enter_nb_proc,0)
            sub_layout_h2.addWidget(self.label_swarmsize,0)
            sub_layout_h2.addWidget(self.enter_swarmsize,0)
            sub_layout_h3.addWidget(self.label_niter,0)
            sub_layout_h3.addWidget(self.enter_niter,0)
            if self.errorweight_index == 1:
                sub_layout_h4.addWidget(self.label_customweight)
                sub_layout_h4.addWidget(self.button_errorFile)
            elif self.errorweight_index ==2:
                sub_layout_h4.addWidget(self.label_customweight)
                sub_layout_h4.addWidget(self.button_errorFile)
            elif self.errorweight_index ==3:
                sub_layout_h4.addWidget(self.label_customweight)
                sub_layout_h4.addWidget(self.button_errorFile)
            self.main_layout=QVBoxLayout()
            self.main_layout.addLayout(sub_layout_h1,0)
            if self.algo_index < 3:
                self.main_layout.addLayout(sub_layout_h2)
            self.main_layout.addLayout(sub_layout_h3)
            if self.errorweight_index == 1:
                self.main_layout.addLayout(sub_layout_h4)
            elif self.errorweight_index == 2:
                self.main_layout.addLayout(sub_layout_h4)
            elif self.errorweight_index == 3:
                self.main_layout.addLayout(sub_layout_h4)
            self.main_layout.addWidget(self.begin_button)
#            self.main_layout.addWidget(self.testguess_button)
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except AttributeError:
                self.setLayout(self.main_layout)
            self.setMaximumHeight(450)
        
        else:
            try:
                deleteLayout(self.layout())
                self.setMaximumHeight(0)
            except:
                print(traceback.format_exc())
                pass
    
    def pressed_loading(self):
        self.controler.loading_text3()
        

        
    def get_weight(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        #print(fileName)
        if self.controler.initialised == 0:
            self.controler.refreshAll3("Please run initialisation first")
            return(0)
        if self.controler.errorIndex != 3:
            try:
                name=os.path.basename(fileName)
                if name =='':
                    self.controler.refreshAll3('Please enter a valid path')
                    return(0)
                errorweight = np.loadtxt(fileName)
                if (len(errorweight) != self.controler.nsample):
                    self.controler.refreshAll3('Please enter a valid path. The file should have {} lines'.format(self.controler.nsample))
                    return(0)
                self.button_errorFile.setText(name)
                self.errorFile = fileName
                self.controler.errorFile = fileName # for graphs
            except:
                print(traceback.format_exc())
                self.controler.refreshAll3('Please enter a valid path')
                return(0)
        else:
            #errorweightdata = h5py.File(fileName, 'r')
            #nameerr = list(errorweightdata.keys())[0]
            #errorweight = list(errorweightdata[nameerr])
            name=os.path.basename(fileName)
            self.button_errorFile.setText(name)
            self.errorFile = fileName
            self.controler.errorFile = fileName # for graphs
            try:
                errorweightdata = h5py.File(fileName, 'r')
                nameerr = list(errorweightdata.keys())[0]
                errorweight = list(errorweightdata[nameerr])
                if np.shape(errorweight)[0] != self.controler.nsample or np.shape(errorweight)[1] != self.controler.nsample:
                    self.refreshAll3('Please enter a valid path. The file should be a {} square matrix'.format(self.controler.nsample))
                    return 0
            except:
                print(traceback.format_exc())
                self.controler.refreshAll3('Please enter a valid path')

            
    def submit_algo_param(self):
        choix_algo=self.algo_index
        if (choix_algo == 3 or choix_algo == 4):
            try:
                from pyOpt import SLSQP
            except:
                print(traceback.format_exc())
                self.controler.refreshAll3("SLSQP was not imported successfully and can't be used")
                return(0)
        swarmsize = 0
        if choix_algo <3: #particle swarm
            try:
                swarmsize=int(self.enter_swarmsize.text())
                if swarmsize<0:
                    self.controler.invalid_swarmsize()
                    return(0)
            except:
                print(traceback.format_exc())
                self.controler.invalid_swarmsize()
                return(0)
        try:
            niter=int(self.enter_niter.text())
            if niter<0:
                self.controler.invalid_niter()
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.invalid_niter()
            return(0)
        self.controler.algo_parameters(choix_algo,swarmsize,niter,self.errorweight_index,self.errorFile)
        
        if self.controler.is_temp_file_1 == 0:
            self.controler.no_temp_file_1()
            return(0)
        if self.controler.is_temp_file_2 == 0:
            self.controler.no_temp_file_2()
            return(0)
        try:
            self.time_domain  = str(self.choices.enter_time_domain.text())
            if self.time_domain=="":
                self.controler.error_message_output_filename()
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.error_message_output_filename()
            return(0)
        try:
            self.frequency_domain = str(self.choices.enter_frequency_domain.text())
            if self.frequency_domain=="":
                self.controler.error_message_output_filename()
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.error_message_output_filename()
            return(0)
        try:
            self.out_opt_filename  = str(self.choices.enter_out_opt.text())
            if self.out_opt_filename=="":
                self.controler.error_message_output_filename()
                return(0)
        except:
            print(traceback.format_exc())
            self.controler.error_message_output_filename()
            return(0)
        try:
            self.controler.get_output_paths(self.choices.outputdir,self.time_domain,
                                            self.frequency_domain, self.out_opt_filename)
        except:
            print(traceback.format_exc())
            self.controler.error_message_output_paths()
            return(0)
        return(1)
                
    def begin_optimization(self):
        global graph_option_2
        submitted = self.submit_algo_param()    #get values from optimisation widget
        if submitted == 1:
            try:
                from mpi4py import MPI
                nb_proc=int(self.enter_nb_proc.text())
            except:
                print(traceback.format_exc())
                self.controler.message_log_tab3("You don't have MPI for parallelization, we'll use only 1 process")
                nb_proc=1
            if self.controler.is_temp_file_5 == 1:
                self.controler.begin_optimization(nb_proc)
                graph_option_2='Pulse (E_field)'
            else:
                self.controler.no_temp_file_5()
    
class FictionalSample_choices(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        
        label_width=200
        action_widget_width=150
        corrective_width_factor=-12
        
        # Widget to see output directory
        self.label_outputdir = QLabel('Directory: ')
        self.label_outputdir.setMaximumWidth(label_width)
        self.button_outputdir = QPushButton('browse', self)
        self.button_outputdir.clicked.connect(self.get_outputdir)
        self.button_outputdir.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.button_outputdir.setMaximumHeight(30)
        
        # Widget to name file
        self.label_name = QLabel('File name: ')
        self.label_name.setMaximumWidth(label_width)
        self.enter_name = QLineEdit()
        self.enter_name.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_name.setMaximumHeight(30)
        
        # Widget to eneter temporal noise standard deviation
        self.label_tempnoise = QLabel('Temporal noise standard deviation (s): ')
        self.label_tempnoise.setMaximumWidth(label_width)
        self.enter_tempnoise = QLineEdit()
        self.enter_tempnoise.setText('0')
        self.enter_tempnoise.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_tempnoise.setMaximumHeight(30)
        
        # Widget to eneter amplitude noise standard deviation
        self.label_ampnoise = QLabel('Amplitude noise standard deviation: ')
        self.label_ampnoise.setMaximumWidth(label_width)
        self.enter_ampnoise = QLineEdit()
        self.enter_ampnoise.setText('0')
        self.enter_ampnoise.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_ampnoise.setMaximumHeight(30)
        
        # Button to create fake sample
        self.fiction_button = QPushButton("Create fictional sample")
        self.fiction_button.clicked.connect(self.generateFictionalSample)
        self.fiction_button.pressed.connect(self.pressed_loading)
        self.fiction_button.setMaximumHeight(20)
        
        # Target permittivity
        self.reference_label = QLabel('Empty cell')
        self.reference_button = QPushButton("browse")
        self.reference_button.setMaximumHeight(20)
        self.reference_button.setMaximumWidth(action_widget_width)
        self.reference_button.clicked.connect(self.search_file)
        
        self.eps_init_button = QPushButton("Compute the permittivity")
        self.eps_init_button.setMaximumHeight(20)
        self.eps_init_button.clicked.connect(self.compute_eps_ini)
        self.eps_init_button.pressed.connect(self.pressed_loading)
        
        # Creation layouts
        main_layout = QVBoxLayout()
        sub_layout_h1=QHBoxLayout()
        sub_layout_h2=QHBoxLayout()
        sub_layout_h3=QHBoxLayout()
        sub_layout_h4=QHBoxLayout()
        sub_layout_h5=QHBoxLayout()
        
        # Organisation layouts for fictional sample
        sub_layout_h1.addWidget(self.label_outputdir)
        sub_layout_h1.addWidget(self.button_outputdir)
        sub_layout_h2.addWidget(self.label_name)
        sub_layout_h2.addWidget(self.enter_name)
        sub_layout_h3.addWidget(self.label_tempnoise)
        sub_layout_h3.addWidget(self.enter_tempnoise)
        sub_layout_h4.addWidget(self.label_ampnoise)
        sub_layout_h4.addWidget(self.enter_ampnoise)
        # Permittivity
        sub_layout_h5.addWidget(self.reference_label)
        sub_layout_h5.addWidget(self.reference_button)
        
        if (self.controler.nlayers == 3)&(self.controler.nfixed_material ==1):
            main_layout.addLayout(sub_layout_h5)
        main_layout.addWidget(self.eps_init_button)
        main_layout.addLayout(sub_layout_h1)
        main_layout.addLayout(sub_layout_h2)
        main_layout.addLayout(sub_layout_h3)
        main_layout.addLayout(sub_layout_h4)
        main_layout.addWidget(self.fiction_button)
        
        self.setLayout(main_layout)
        
    def pressed_loading(self):
        self.controler.loading_text3()
    def compute_eps_ini(self, nbpi):
        self.nbpi=0
        self.controler.compute_eps_init(self.emptycellfile, self.nbpi)

    def generateFictionalSample(self): #files
        name = self.enter_name.text()
        try:
            name  = str(self.enter_name.text())
            if name=="":
                self.controler.refreshAll3("Please enter a valid name")
        except:
            print(traceback.format_exc())
            self.controler.refreshAll3("Please enter a valid name")
        try:
            directory = self.outputdir
        except AttributeError:
            directory = None
        tempstd = float(self.enter_tempnoise.text())
        ampstd  = float(self.enter_ampnoise.text())
        self.controler.generateFictionalSample(tempstd,ampstd,name,directory)
    
    
    def get_outputdir(self):
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.outputdir=str(DirectoryName)
            name=os.path.basename(str(DirectoryName))
            self.button_outputdir.setText(name)
        except:
            print(traceback.format_exc())
            self.controler.error_message_path3()
    def search_file(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        try:
            name=os.path.basename(fileName)
            if name =='':
                self.controler.refreshAll3('Please enter a valid path')
                return(0)
            self.reference_button.setText(name)
            self.emptycellfile = fileName
        except:
            print(traceback.format_exc())
            self.controler.refreshAll3('Please enter a valid path')
            return(0)


class log_optimisation(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient3(self)
        self.setReadOnly(True)

        references=[to_sup(1)+" Error options:",
                    "Constant weight: error = \u2016Efit(w)-E(w)\u2016 / \u2016E(w)\u2016",
                    "or error = \u2016Efit(t)-E(t)\u2016 / \u2016E(t)\u2016 in super resolution",
                    "Custom weight: error \u03B1 \u2016(Efit(t)-E(t))*weight \u2016 / \u2016E(t)\u2016 ",
                    "Custom noise: error \u03B1 \u2016(Efit(t)-E(t))/noise \u2016 / \u2016E(t)\u2016 ",
                    "Custom noise matrix: error ..."]
        for i in references:
            self.append(i)

    def refresh(self):
        message = self.controler.message
        if type(message)==list:
            for i in message:
                self.append(i)
        else:
            self.append(message)

class Graphs_optimisation(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient3(self)
        self.setTitle("Graphs")
        # Create objects to plot graphs
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.draw()
        # Create buttons to chose what to plot
        # Real part of refractive index
        self.button_real_index = QPushButton('Real(n)', self)
        self.button_real_index.clicked.connect(self.real_index_graph)
        # Imaginary part of refractive index
        self.button_im_index = QPushButton('Im(n)', self)
        self.button_im_index.clicked.connect(self.im_index_graph)
        # E field
        self.button_E_field = QPushButton('E field', self)
        self.button_E_field.clicked.connect(self.E_field_graph)
        # E field [dB]
        self.button_E_field_dB = QPushButton('E field  [dB]', self)
        self.button_E_field_dB.clicked.connect(self.E_field_dB_graph)
        # Pulse (E field)
        self.button_Pulse_E_field = QPushButton('Pulse E field', self)
        self.button_Pulse_E_field.clicked.connect(self.Pulse_E_field_graph)
        # Pulse (E field) [dB]
        self.button_Pulse_E_field_dB = QPushButton('Pulse E field [dB]', self)
        self.button_Pulse_E_field_dB.clicked.connect(self.Pulse_E_field_dB_graph)
        # frequency filter
        self.button_freq_filter = QPushButton('Frequency filter')
        self.button_freq_filter.clicked.connect(self.freq_filter_graph)
        # temporal filter
        self.button_temp_filter = QPushButton('Temporal filter')
        self.button_temp_filter.clicked.connect(self.temp_filter_graph)
        # E field residue
        self.button_E_field_residue = QPushButton('E field residue', self)
        self.button_E_field_residue.clicked.connect(self.E_field_residue_graph)
        # E field residue [dB]
        self.button_E_field_residue_dB = QPushButton('E field residue [dB]', self)
        self.button_E_field_residue_dB.clicked.connect(self.E_field_residue_dB_graph)
        # Pulse residue (E field)
        self.button_Pulse_E_field_residue = QPushButton('E(t) residue', self)
        self.button_Pulse_E_field_residue.clicked.connect(self.Pulse_E_field_residue_graph)
        # Pulse residue (E field) [dB]
        self.button_Pulse_E_field_residue_dB = QPushButton('E(t) residue [dB]', self)
        self.button_Pulse_E_field_residue_dB.clicked.connect(self.Pulse_E_field_residue_dB_graph)


        # Organisation layout
        self.vlayoutmain = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.hlayout2 = QHBoxLayout()
        self.hlayout.addWidget(self.button_real_index)
        self.hlayout.addWidget(self.button_im_index)
        self.hlayout.addWidget(self.button_E_field)
        self.hlayout.addWidget(self.button_E_field_dB)
        self.hlayout.addWidget(self.button_Pulse_E_field)
        self.hlayout.addWidget(self.button_Pulse_E_field_dB)
        self.hlayout2.addWidget(self.button_freq_filter)
        self.hlayout2.addWidget(self.button_temp_filter)
        self.hlayout2.addWidget(self.button_E_field_residue)
        self.hlayout2.addWidget(self.button_E_field_residue_dB)
        self.hlayout2.addWidget(self.button_Pulse_E_field_residue)
        self.hlayout2.addWidget(self.button_Pulse_E_field_residue_dB)
        self.vlayoutmain.addWidget(self.toolbar)
        self.vlayoutmain.addWidget(self.canvas)
        self.vlayoutmain.addLayout(self.hlayout)
        self.vlayoutmain.addLayout(self.hlayout2)
        self.setLayout(self.vlayoutmain)
#        self.drawgraph()


    def draw_graph_init(self,myinputdata, myreferencedata, myfitteddata, epsilonTarget,
                        myglobalparameters, freqWindow, timeWindow, weight, noise, noisematrix, errorweight_index):
        global graph_option_2
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        if graph_option_2=='Real(refractive index)':
            ax1.set_title('Real part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Real part of refractive index',color=color)
            if epsilonTarget is not None:
                ax1.plot(myglobalparameters.freq, np.sqrt(epsilonTarget).real, 'b-', label='target')
            ax1.plot(myglobalparameters.freq, np.sqrt(myfitteddata.epsilon[0]).real, 'g-', label='fited')
            ax1.legend()

        elif graph_option_2=='Im(refractive index)':
            ax1.set_title('Imaginary part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Imaginary part of refractive index',color=color)
            if epsilonTarget is not None:
                ax1.plot(myglobalparameters.freq, np.sqrt(epsilonTarget).imag, 'r-', label='target')
            ax1.plot(myglobalparameters.freq, -np.sqrt(myfitteddata.epsilon[0]).imag, 'g-', label='fited')
            ax1.legend()

        elif graph_option_2=='E_field':
            ax1.set_title('E_field', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field',color=color)
            ax1.plot(myglobalparameters.freq,(abs(myreferencedata.Spulseinit)), 'g-', label='reference spectre (log)')
            ax1.plot(myglobalparameters.freq,(abs(myinputdata.Spulse)), 'b-', label='spectre after sample (log)')
            ax1.plot(myglobalparameters.freq,(abs(myfitteddata.Spulse)), 'r-', label='fited spectre (log)')
            ax1.legend()

        elif graph_option_2=='E_field [dB]':
            ax1.set_title('E_field [dB]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field [dB]',color=color)
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myreferencedata.Spulseinit))/np.log(10), 'g-', label='reference spectre (log)')
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myinputdata.Spulse))/np.log(10), 'b-', label='spectre after (log)')
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myfitteddata.Spulse))/np.log(10), 'r-', label='fited spectre (log)')

            ax1.legend()

        elif graph_option_2=='Pulse (E_field)':
            ax1.set_title('Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field)',color=color)
            ax1.plot(myglobalparameters.t, myreferencedata.Pulseinit, 'g-', label='reference pulse')
            ax1.plot(myglobalparameters.t, myinputdata.pulse, 'b-', label='pulse after sample')
            ax1.plot(myglobalparameters.t, myfitteddata.pulse, 'r-', label='fited pulse')
            if weight is not None:
                ax1.plot(myglobalparameters.t, weight, 'c-', label='normalised weight')
                ax1.plot(myglobalparameters.t, -weight, 'c-')
            elif noise is not None:
                ax1.plot(myglobalparameters.t, noise, 'c-', label='normalised noise')
                ax1.plot(myglobalparameters.t, -noise, 'c-')
            ax1.legend()

        elif graph_option_2 == 'Pulse (E_field) [dB]':
            ax1.set_title('Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field)',color=color)
            ax1.plot(myglobalparameters.t, 10*np.log(myreferencedata.Pulseinit**2)/np.log(10), 'g-', label='reference pulse')
            ax1.plot(myglobalparameters.t, 10*np.log(myinputdata.pulse**2)/np.log(10), 'b-', label='pulse after sample')
            ax1.plot(myglobalparameters.t, 10*np.log(myfitteddata.pulse**2)/np.log(10), 'r-', label='pulse fited')
            ax1.legend()

        elif graph_option_2=='freq_filter':
            ax1.set_title('Frequency filter', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Frequency filter',color=color)
            ax1.plot(myglobalparameters.freq, freqWindow, 'g-', label='filter')
            ax1.legend()
        
        elif graph_option_2=='temp_filter':
            ax1.set_title('Temporal filter', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Temporal filter',color=color)
            ax1.plot(myglobalparameters.t, timeWindow, 'g-', label='filter')
            ax1.legend()

        elif graph_option_2=='E_field residue':
            ax1.set_title('E_field residue', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field residue',color=color)
            if errorweight_index==3:
                ax1.plot(myglobalparameters.freq,(abs(np.fft.rfft(np.dot(noisematrix,(myinputdata.pulse-myfitteddata.pulse))))), 'r-', label='[spectre after sample] - [fited spectre] normalized by noisematrix')
            else:
                ax1.plot(myglobalparameters.freq,(abs(myinputdata.Spulse-myfitteddata.Spulse)), 'r-', label='[spectre after sample] - [fited spectre]')
            ax1.legend()

        elif graph_option_2=='E_field residue [dB]':
            ax1.set_title('E_field residue [dB]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field residue [dB]',color=color)
            if errorweight_index==3:
                ax1.plot(myglobalparameters.freq,(20*np.log(abs(np.fft.rfft(np.dot(noisematrix,(myinputdata.pulse-myfitteddata.pulse))))))/np.log(10), 'r-', label='[spectre after sample] - [fited spectre] normalized by noisematrix (log)')
            else:
                ax1.plot(myglobalparameters.freq,20*np.log(abs(myinputdata.Spulse-myfitteddata.Spulse))/np.log(10), 'r-', label='[spectre after sample] - [fited spectre (log)]')

            ax1.legend()

        elif graph_option_2=='Pulse (E_field) residue':
            ax1.set_title('Pulse (E_field) residue', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field) residue',color=color)
            if errorweight_index == 3:
                ax1.plot(myglobalparameters.t, np.dot(noisematrix,(myinputdata.pulse-myfitteddata.pulse)), 'b-', label='[pulse after sample] - [fited pulse] normalized by noisematrix')
            else:
                ax1.plot(myglobalparameters.t, myinputdata.pulse-myfitteddata.pulse, 'b-', label='[pulse after sample] - [fited pulse]')
            ax1.legend()

        else:
            ax1.set_title('Pulse (E_field) residue [dB]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field) residue [dB]',color=color)
            if errorweight_index==3:
                ax1.plot(myglobalparameters.t, 20*np.log(np.dot(noisematrix,(myinputdata.pulse-myfitteddata.pulse)))/np.log(10), 'b-', label='[pulse after sample] - [fited pulse] normalized by noisematrix (log)')
            else:
                ax1.plot(myglobalparameters.t, 20*np.log(myinputdata.pulse-myfitteddata.pulse)/np.log(10), 'b-', label='[pulse after sample] - [fited pulse] (log)')
            ax1.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def real_index_graph(self):
        global graph_option_2
        graph_option_2='Real(refractive index)'
        self.controler.ploting_text3('Ploting real part of refractive index')

    def im_index_graph(self):
        global graph_option_2
        graph_option_2='Im(refractive index)'
        self.controler.ploting_text3('Ploting imaginary part of refractive index')

    def E_field_graph(self):
        global graph_option_2
        graph_option_2='E_field'
        self.controler.ploting_text3('Ploting E_field')

    def E_field_dB_graph(self):
        global graph_option_2
        graph_option_2='E_field [dB]'
        self.controler.ploting_text3('Ploting E_field [dB]')

    def Pulse_E_field_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field)'
        self.controler.ploting_text3('Ploting pulse E_field')

    def Pulse_E_field_dB_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) [dB]'
        self.controler.ploting_text3('Ploting pulse E_field [dB]')
        
    def freq_filter_graph(self):
        global graph_option_2
        graph_option_2='freq_filter'
        self.controler.ploting_text3('Ploting frequency filter')

    def temp_filter_graph(self):
        global graph_option_2
        graph_option_2='temp_filter'
        self.controler.ploting_text3('Ploting temporal filter')

    def E_field_residue_graph(self):
        global graph_option_2
        graph_option_2='E_field residue'
        self.controler.ploting_text3('Ploting E_field residue')

    def E_field_residue_dB_graph(self):
        global graph_option_2
        graph_option_2='E_field residue [dB]'
        self.controler.ploting_text3('Ploting E_field residue [dB]')

    def Pulse_E_field_residue_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) residue'
        self.controler.ploting_text3('Ploting pulse E_field residue')

    def Pulse_E_field_residue_dB_graph(self):
        global graph_option_2
        graph_option_2='Pulse (E_field) residue [dB]'
        self.controler.ploting_text3('Ploting pulse E_field residue [dB]')

    def refresh(self):
        try:
            epsilonTarget=self.controler.epsilonTarget
            myinputdata=self.controler.myinputdata
            myreferencedata=self.controler.myreferencedata
            if self.controler.myfitteddata != None: 
                myfitteddata=self.controler.myfitteddata
            else:
                myfitteddata=self.controler.previewdata
            myglobalparameters=self.controler.myglobalparameters
            freqWindow = self.controler.Freqwindow
            timeWindow = self.controler.timeWindow
            weight = self.controler.normalisedWeight
            noise = self.controler.normalisedNoise
            noisematrix = self.controler.noisematrix
            errorweight_index = self.controler.errorIndex
            self.draw_graph_init(myinputdata,myreferencedata,myfitteddata,
                                 epsilonTarget,myglobalparameters,freqWindow,
                                 timeWindow, weight, noise, noisematrix, errorweight_index)
        except:
            print(traceback.format_exc())
            pass



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class MainWindow(QMainWindow):
    def __init__(self, controler):
        super().__init__()
        self.setWindowTitle("Fit@TDS")
        self.mainwidget = MyTableWidget(self,controler)
        self.setCentralWidget(self.mainwidget)

    def closeEvent(self,event):
        try:
            shutil.rmtree("temp")
        except:
            print(traceback.format_exc())
            pass

def main():
    app = QApplication([])
    controler = Controler()
    win = MainWindow(controler)
    controler.init()
    # win.show()
    win.showMaximized()
    app.exec()

if __name__ == '__main__':
    main()
