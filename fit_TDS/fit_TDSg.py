#!/usr/bin/python
# -*- coding: latin-1 -*-

import sys
import random
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
try:
    import shutil
except:
    pass

from fit_TDSc import Controler


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    size = comm.Get_size()
except:
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


graph_option='Transmission'
graph_option_2=None



class MyTableWidget(QWidget):

    def __init__(self, parent,controler):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tab1 = Initialisation_tab(self,controler)
        self.tab2 = Model_parameters_tab(self,controler)
        self.tab3 = Optimization_tab(self,controler)

        # Add tabs
        self.tabs.addTab(self.tab1,"Initialisation")
        self.tabs.addTab(self.tab2,"Model parameters")
        self.tabs.addTab(self.tab3,"Optimization")

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


###############################################################################
###############################################################################
#######################   Initialisation tab   ################################
###############################################################################
###############################################################################

class Initialisation_tab(QWidget):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        hlayout = QHBoxLayout()
        vlayout1 = QVBoxLayout()
        vlayout2 = QVBoxLayout()
        self.init_param_widget = InitParamWidget(self, controler)
        self.text_box = TextBoxWidget(self, controler)
        self.chart_widget = ChartWidget(self, controler)
        vlayout1.addWidget(self.init_param_widget, 0)
        vlayout1.addWidget(self.text_box, 1)
        vlayout2.addWidget(self.chart_widget, 0)
        hlayout.addLayout(vlayout1, 0)
        hlayout.addLayout(vlayout2, 1)
        self.setLayout(hlayout)


class InitParamWidget(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.setTitle("Parameters values")
#        self.setFixedWidth(300)
        # We create the text associated to the text box
        self.label_thickness = QLabel('Thickness of the sample (m)')
        self.label_thickness.setAlignment(Qt.AlignVCenter)
        self.label_thickness.resize(200, 100);
        self.label_thickness.resize(self.label_thickness.sizeHint());

        self.label_uncertainty = QLabel('Uncertainty of the thickness (%)')
        self.label_uncertainty.setAlignment(Qt.AlignVCenter)
        self.label_uncertainty.resize(200, 100);
        self.label_uncertainty.resize(self.label_uncertainty.sizeHint());

        self.label_path_without_sample = QLabel('Select data (without sample)')
        self.label_path_without_sample.setAlignment(Qt.AlignVCenter)
        self.label_path_without_sample.resize(200, 100);
        self.label_path_without_sample.resize(self.label_path_without_sample.sizeHint());

        self.label_path_with_sample = QLabel('Select data (with sample)')
        self.label_path_with_sample.setAlignment(Qt.AlignVCenter)
        self.label_path_with_sample.resize(200, 100);
        self.label_path_with_sample.resize(self.label_path_with_sample.sizeHint());

        # We create text box for the user to enter values of the sample
        self.Thickness_box=QLineEdit(self)
        self.Thickness_box.setAlignment(Qt.AlignVCenter)
        self.Thickness_box.resize(200, 100);
        self.Thickness_box.resize(self.Thickness_box.sizeHint());

        self.Uncertainty_box=QLineEdit(self)
        self.Uncertainty_box.setAlignment(Qt.AlignVCenter)
        self.Uncertainty_box.resize(200, 100);
        self.Uncertainty_box.resize(self.Uncertainty_box.sizeHint());

        self.button_ask_path_without_sample = QPushButton('browse', self)
#        self.button_ask_path_without_sample.setLayoutDirection(Qt.RightToLeft)
        self.button_ask_path_without_sample.resize(200, 100);
        self.button_ask_path_without_sample.resize(self.button_ask_path_without_sample.sizeHint());
        self.button_ask_path_without_sample.clicked.connect(self.get_path_without_sample)

        self.button_ask_path_with_sample = QPushButton('browse', self)
#        self.button_ask_path_with_sample.setLayoutDirection(Qt.RightToLeft)
        self.button_ask_path_with_sample.resize(200, 100);
        self.button_ask_path_with_sample.resize(self.button_ask_path_with_sample.sizeHint());
        self.button_ask_path_with_sample.clicked.connect(self.get_path_with_sample)

        # We create a button to extract the information from the text boxes
        self.button = QPushButton('Submit', self)
        self.button.clicked.connect(self.on_click)
        self.button.pressed.connect(self.pressed_loading)

        # Organisation layout
        self.hlayout1=QHBoxLayout()
        self.hlayout2=QHBoxLayout()
        self.hlayout3=QHBoxLayout()
        self.hlayout4=QHBoxLayout()
        self.vlayoutmain=QVBoxLayout()

        self.hlayout1.addWidget(self.label_thickness,1)
        self.hlayout1.addWidget(self.Thickness_box,0)

        self.hlayout2.addWidget(self.label_uncertainty,1)
        self.hlayout2.addWidget(self.Uncertainty_box,0)

        self.hlayout3.addWidget(self.label_path_without_sample,20)
        self.hlayout3.addWidget(self.button_ask_path_without_sample,17)

        self.hlayout4.addWidget(self.label_path_with_sample,20)
        self.hlayout4.addWidget(self.button_ask_path_with_sample,17)

        self.vlayoutmain.addLayout(self.hlayout1)
        self.vlayoutmain.addLayout(self.hlayout2)
        self.vlayoutmain.addLayout(self.hlayout3)
        self.vlayoutmain.addLayout(self.hlayout4)
        self.vlayoutmain.addWidget(self.button)
        self.setLayout(self.vlayoutmain)


    def pressed_loading(self):
        self.controler.loading_text()

    def on_click(self):
        try:
            thickness=float(self.Thickness_box.text())
            uncertainty=float(self.Uncertainty_box.text())
            if thickness<0:
                self.controler.warning_negative_thickness()
                return(0)
            if uncertainty<0 or uncertainty>100:
                self.controler.warning_uncertainty()
                return(0)
            try:
                self.controler.param_ini(thickness,uncertainty,self.path_without_sample,self.path_with_sample)
            except:
                self.controler.error_message_path()
        except:
            self.controler.error_message_init_values()
        self.controler.is_thickness = 1


    def get_path_without_sample(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Without sample", options=options)
        try:
            self.path_without_sample=fileName
            name=self.controler.name_file(fileName)
            self.button_ask_path_without_sample.setText(name)
        except:
            self.controler.error_message_path()

    def get_path_with_sample(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"With sample", options=options)
        try:
            self.path_with_sample=fileName
            name=self.controler.name_file(fileName)
            self.button_ask_path_with_sample.setText(name)
        except:
            self.controler.error_message_path()

    def refresh(self):
        pass


class TextBoxWidget(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
#        self.setFixedHeight(150)
        self.setReadOnly(True)
        self.append("Log")

    def refresh(self):
        message = self.controler.message
        if message:
            self.append(message)

class ChartWidget(QGroupBox):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient(self)
        self.setTitle("Graphs")
        # Create objects to plot graphs
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.draw()
        # Create buttons to chose what to plot
        # Transmission
        self.button_transmission = QPushButton('Transmission', self)
        self.button_transmission.clicked.connect(self.transmission_graph)
        # Permitivity
        self.button_Permitivity = QPushButton('Permitivity', self)
        self.button_Permitivity.clicked.connect(self.Permitivity_graph)
        # Losses
        self.button_Losses = QPushButton('Losses', self)
        self.button_Losses.clicked.connect(self.Losses_graph)
        # Spectral density of energy [dB]
        self.button_Spectral = QPushButton('Spectral', self)
        self.button_Spectral.clicked.connect(self.Spectral_graph)

        # Organisation layout
        self.vlayoutmain = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.button_transmission)
        self.hlayout.addWidget(self.button_Permitivity)
        self.hlayout.addWidget(self.button_Losses)
        self.hlayout.addWidget(self.button_Spectral)
        self.vlayoutmain.addWidget(self.canvas)
        self.vlayoutmain.addLayout(self.hlayout)
        self.setLayout(self.vlayoutmain)
#        self.drawgraph()


    def draw_graph_init(self,monepsilon,myinputdata,myinputdatafromfile,z,myglobalparameters):
        global graph_option
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        if graph_option=='Transmission':
            ax1.set_title('Transmission', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Transmission',color=color)
            ax1.plot(myglobalparameters.freq, abs(myinputdata.Spulse/myinputdatafromfile.Spulseinit),color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        elif graph_option=='Permitivity':
            ax1.set_title('Permittivity', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Permitivity',color=color)
            ax1.plot(myglobalparameters.freq,  np.real((myinputdata.epsilon)), 'b-', label='real part')
            ax1.plot(myglobalparameters.freq, np.imag((myinputdata.epsilon)), 'r-', label='imaginary part')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend()

        elif graph_option=='Losses':
            ax1.set_title('Losses [1/m]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Losses [1/m]',color=color)
            ax1.plot(myglobalparameters.freq, np.log(abs(myinputdata.Spulse/myinputdatafromfile.Spulseinit))/z,color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        else:
            ax1.set_title('Spectral density of energy', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Spectral density of energy [dB]',color=color)
            ax1.plot(myglobalparameters.freq, 20*np.log(abs(myinputdata.Spulse))/np.log(10), 'b-', label='with sample')
            ax1.plot(myglobalparameters.freq, 20*np.log(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='initial pulse')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def transmission_graph(self):
        global graph_option
        graph_option='Transmission'
        self.controler.ploting_text('Ploting transmission')

    def Permitivity_graph(self):
        global graph_option
        graph_option='Permitivity'
        self.controler.ploting_text('Ploting Permitivity')

    def Losses_graph(self):
        global graph_option
        graph_option='Losses'
        self.controler.ploting_text('Ploting Losses')

    def Spectral_graph(self):
        global graph_option
        graph_option='Spectral'
        self.controler.ploting_text('Ploting Spectral density of energy')

    def refresh(self):
        try:
            monepsilon=self.controler.monepsilon
            myinputdata=self.controler.myinputdata
            myinputdatafromfile=self.controler.myinputdatafromfile
            z=self.controler.z
            myglobalparameters=self.controler.myglobalparameters
            self.draw_graph_init(monepsilon,myinputdata,myinputdatafromfile,z,myglobalparameters)
        except:
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

        # Creation widgets
        label_width=200
        action_widget_width=200
        corrective_width_factor=-12
        # Algorithm choice
        self.label_algo = QLabel("Algorithm \u00b9")
        self.label_algo.setMaximumWidth(label_width)
        self.options_algo = QComboBox()
        self.options_algo.addItems(['NumPy optimize swarm particle',
                                    'ALPSO without parallelization',
                                    'ALPSO with parallelization'])
        self.options_algo.setMaximumWidth(action_widget_width)
        # Photonic structre
        self.label_struct = QLabel("Photonic structure \u00b2")
        self.label_struct.setMaximumWidth(label_width)
        self.options_struct = QComboBox()
        self.options_struct.addItems(['Transmission Fabry-Perot',
                                      'Transmission Fabry-Perot \n with a resonator (TDCMT)'])
        self.options_struct.setMaximumWidth(action_widget_width +
                                            corrective_width_factor)
        self.options_struct.setMaximumHeight(30)
        # Thickness
        self.label_tickness = QLabel("Fit thickness \u00b3")
        self.label_tickness.setMaximumWidth(label_width)
        self.options_tickness = QComboBox()
        self.options_tickness.addItems(['Yes', 'No'])
        self.options_tickness.setMaximumWidth(action_widget_width)
        # Drude term
        self.label_drude = QLabel("Drude term \u2074")
        self.label_drude.setMaximumWidth(label_width)
        self.options_drude = QComboBox()
        self.options_drude.addItems(['Yes', 'No'])
        self.options_drude.setMaximumWidth(action_widget_width)
        # Scattering
        self.label_scattering = QLabel("Scattering \u2075")
        self.label_scattering.setMaximumWidth(label_width)
        self.options_scattering = QComboBox()
        self.options_scattering.addItems(['Yes', 'No'])
        self.options_scattering.setMaximumWidth(action_widget_width)
        # Lorentz Oscillators
        self.label_lorentz = QLabel("Number of Lorentz \noscillators \u2076")
        self.label_lorentz.setMaximumWidth(label_width)
        self.enter_lorentz = QLineEdit()
        self.enter_lorentz.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_lorentz.setMaximumHeight(30)

        # Debye Oscillators
        self.label_debye = QLabel("Number of Debye \noscillators \u2077")
        self.label_debye.setMaximumWidth(label_width)
        self.enter_debye = QLineEdit()
        self.enter_debye.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_debye.setMaximumHeight(30)

        # SwarmSize
        self.label_swarmsize = QLabel("Swarmsize \u2078")
        self.label_swarmsize.setMaximumWidth(label_width)
        self.enter_swarmsize = QLineEdit()
        self.enter_swarmsize.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_swarmsize.setMaximumHeight(30)

        # Number of iterations
        self.label_niter = QLabel("Number of iterations \u2079")
        self.label_niter.setMaximumWidth(label_width)
        self.enter_niter = QLineEdit()
        self.enter_niter.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_niter.setMaximumHeight(30)



        # OK button
        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(self.submit_model_param)


        # Creation layouts
        main_layout=QVBoxLayout()
        sub_layout_h1=QHBoxLayout()
        sub_layout_h2=QHBoxLayout()
        sub_layout_h3=QHBoxLayout()
        sub_layout_h4=QHBoxLayout()
        sub_layout_h5=QHBoxLayout()
        sub_layout_h6=QHBoxLayout()
        sub_layout_h7=QHBoxLayout()
        sub_layout_h8=QHBoxLayout()
        sub_layout_h9=QHBoxLayout()

        # Organisation Layouts
        sub_layout_h1.addWidget(self.label_algo,0)
        sub_layout_h1.addWidget(self.options_algo,0)

        sub_layout_h2.addWidget(self.label_struct,0)
        sub_layout_h2.addWidget(self.options_struct,0)

        sub_layout_h3.addWidget(self.label_tickness,0)
        sub_layout_h3.addWidget(self.options_tickness,0)

        sub_layout_h4.addWidget(self.label_drude,0)
        sub_layout_h4.addWidget(self.options_drude,0)

        sub_layout_h5.addWidget(self.label_scattering,0)
        sub_layout_h5.addWidget(self.options_scattering,0)

        sub_layout_h6.addWidget(self.label_lorentz,0)
        sub_layout_h6.addWidget(self.enter_lorentz,0)

        sub_layout_h7.addWidget(self.label_debye,0)
        sub_layout_h7.addWidget(self.enter_debye,0)

        sub_layout_h8.addWidget(self.label_swarmsize,0)
        sub_layout_h8.addWidget(self.enter_swarmsize,0)

        sub_layout_h9.addWidget(self.label_niter,0)
        sub_layout_h9.addWidget(self.enter_niter,0)

        main_layout.addLayout(sub_layout_h1)
        main_layout.addLayout(sub_layout_h2)
        main_layout.addLayout(sub_layout_h3)
        main_layout.addLayout(sub_layout_h4)
        main_layout.addLayout(sub_layout_h5)
        main_layout.addLayout(sub_layout_h6)
        main_layout.addLayout(sub_layout_h7)
        main_layout.addLayout(sub_layout_h8)
        main_layout.addLayout(sub_layout_h9)
        main_layout.addWidget(self.button_submit)

        self.setLayout(main_layout)

    def submit_model_param(self):
        choix_algo=self.options_algo.currentIndex()
        mymodelstruct=self.options_struct.currentIndex()
        thickness=self.options_tickness.currentIndex()
        isdrude=self.options_drude.currentIndex()
        scattering=self.options_scattering.currentIndex()
        try:
            n=int(self.enter_lorentz.text())
        except:
            self.controler.invalid_n_lorentz()
        try:
            nDebye=int(self.enter_debye.text())
        except:
            self.controler.invalid_n_debye()
        try:
            swarmsize=int(self.enter_swarmsize.text())
        except:
            self.controler.invalid_swarmsize()
        try:
            niter=float(self.enter_niter.text())
        except:
            self.controler.invalid_niter()
        try:
            if self.controler.is_thickness == 1:
                self.controler.parameters_values(choix_algo,mymodelstruct,thickness,isdrude,scattering,n,nDebye,swarmsize,niter)
            else:
                self.controler.invalid_tun_opti_first()
                return(0)
        except:
            self.controler.invalid_param_opti()

    def refresh(self):
        pass


class references_widget(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
        self.setFixedHeight(200)
        self.setFixedWidth(400)
        # self.setReadOnly(True)
        self.append("References")
        references=['\u00b9 Choice of algorithm for optimization. \n',
                    '\u00b2 Choice to set a model for the photonic structure. \n',
                    '\u00b3 Set the thickness of the sample as an optimization parameter. \n',
                    '\u2074 Drude model depicts the permitivity Epsillon as Eps =Eps_0- Omega_p^2/(Omega^2-j*gamma*omega). \n',
                    '\u2075 Do you want to take into account scattering ?'
                    '\u2076 Lorentz model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon*Omega_0^2]/[Omega_0^2+j*gamma*Omega-Omega^2]. \n'
                    '\u2077 Debye model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon/[1+j*Omega/OmegaD]. \n',
                    '\u2078 Swarmsize for optimization. \n',
                    '\u2079 Number of iterations for optimization. \n']
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

        parameters_values=parameters_values_scroll(self,self.controler)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(parameters_values)

        self.main_layout=QVBoxLayout()
        self.main_layout.addWidget(scroll)
#        self.main_layout.setAlignment(Qt.AlignCenter)
        if nb_param:
            try:
                deleteLayout(self.layout())
                self.layout().deleteLater()
            except:
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

        # Path of a file containing tha values of the parameters (optional)
        self.path_param_values=None

        label_width=1500
        text_box_width=100
        text_box_height=25


        # Creation Widgets et Layouts
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
        layouts=[]

        self.text_boxes_value=[]
        self.text_boxes_min=[]
        self.text_boxes_max=[]

        self.search_path_button = QPushButton("...")
        self.search_path_button.clicked.connect(self.search_path)
        self.search_path_button.setMaximumWidth(text_box_width+10)
        self.search_path_button.setMaximumHeight(text_box_height)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_values)
        self.submit_button.setMaximumWidth(text_box_width)
        self.submit_button.setMaximumHeight(text_box_height)

        self.log_box=log_param_values(self,self.controler)
        self.log_box.setMaximumWidth(400)
        self.log_box.setMaximumHeight(50)


        if nb_param:
            for i in range(nb_param):
                self.labels.append(QLabel(f'{mydescription[i]} ({myunits[i]})'))
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
        self.main_layout.addLayout(sub_layout_h2)

        sub_layout_h3.addWidget(self.log_box)
        sub_layout_h3.addWidget(self.submit_button)
        self.main_layout.addLayout(sub_layout_h3)

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
                self.log_box.append("Values not submited ! There is a problem with the values and intervals of the parameters.")
                return(0)
        except:
            self.log_box.append("Invalid values.")
            return(0)
        self.controler.mesparam=mesparam
        self.controler.save_optimisation_param(self.controler.mesparam)
        self.log_box.append("Values submitted")

    def search_path(self):
        if self.controler.is_thickness == 1:
            nb_param=self.controler.nb_param
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
            try:
                self.path_param_values=fileName
                name=self.controler.name_file(fileName)
                self.search_path_button.setText(name)
                self.log_box.append("Values taken from "+name)
            except:
                self.controler.error_message_path()

            self.controler.mesparam=np.loadtxt(self.path_param_values)
            mes_param=self.controler.mesparam

            try:
                if nb_param==len(mes_param[:,0]):
                    for i in range(nb_param):
                        self.text_boxes_value[i].setText(f'{mes_param[i,0]:.3E}')
                        self.text_boxes_min[i].setText(f'{mes_param[i,1]:.3E}')
                        self.text_boxes_max[i].setText(f'{mes_param[i,2]:.3E}')
                else:
                    self.log_box.append("The file submitted does not have the same number of parameters as the model chosen.")
                    return(0)
            except:
                self.log_box.append("There is a problem with the file submitted.")
                return(0)
        else:
            self.log_box.append("Please run the initialisation window first.")

class log_param_values(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient2(self)
#        self.setFixedHeight(150)
        self.setReadOnly(True)
        self.append("Optimisation")

    def message_log(self,message):
        self.append(message)
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
        self.log_widget=log_optimisation(self,controler)

        self.graph_widget=Graphs_optimisation(self,controler)

        # Corrective factors
        label_width=200
        action_widget_width=150
        corrective_width_factor=-12

        # Button to launch optimization
        self.begin_button = QPushButton("Begin")
        self.begin_button.clicked.connect(self.begin_optimization)
        self.begin_button.pressed.connect(self.pressed_loading)
        self.begin_button.setMaximumWidth(50)
        self.begin_button.setMaximumHeight(20)

        # Preview button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.preview_button.pressed.connect(self.pressed_loading)
        self.preview_button.setMaximumWidth(70)
        self.preview_button.setMaximumHeight(20)

        # Wiget to see how many process are going to be used for the omptimization
        self.label_nb_proc = QLabel("How many process do you want to use?")
        self.enter_nb_proc = QLineEdit()
        self.enter_nb_proc.setText('1')
        self.enter_nb_proc.setMaximumWidth(50)
        self.enter_nb_proc.setMaximumHeight(25)

        # Widget to see output directory
        self.label_outputdir = QLabel('Output directory: ')
        self.label_outputdir.setMaximumWidth(label_width)
        self.button_outputdir = QPushButton('browse', self)
        self.button_outputdir.clicked.connect(self.get_outputdir)
        self.button_outputdir.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.button_outputdir.setMaximumHeight(30)


        # Widget to name time domain output
        self.label_time_domain = QLabel('Time domain output filename: ')
        self.label_time_domain.setMaximumWidth(label_width)
        self.enter_time_domain = QLineEdit()
        self.enter_time_domain.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_time_domain.setMaximumHeight(30)


        # Widget to name frequency domain output
        self.label_frequency_domain = QLabel('Frequency domain output filename: ')
        self.label_frequency_domain.setMaximumWidth(label_width)
        self.enter_frequency_domain = QLineEdit()
        self.enter_frequency_domain.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_frequency_domain.setMaximumHeight(30)


        # Widget to name optimization output
        self.label_out_opt = QLabel('Ouput optimization: ')
        self.label_out_opt.setMaximumWidth(label_width)
        self.enter_out_opt = QLineEdit()
        self.enter_out_opt.setMaximumWidth(action_widget_width +
                                           corrective_width_factor)
        self.enter_out_opt.setMaximumHeight(30)


        # Creation layouts
        main_layout = QHBoxLayout()
        sub_layout_v=QVBoxLayout()
        sub_layout_h=QHBoxLayout()
        sub_layout_h_2=QHBoxLayout()
        sub_layout_h_3=QHBoxLayout()
        sub_layout_h_4=QHBoxLayout()
        sub_layout_h_5=QHBoxLayout()
        sub_layout_h_6=QHBoxLayout()


        # Organisation layouts
        sub_layout_h.addWidget(self.label_nb_proc)
        sub_layout_h.addWidget(self.enter_nb_proc)
        sub_layout_h_2.addWidget(self.label_outputdir)
        sub_layout_h_2.addWidget(self.button_outputdir)
        sub_layout_h_3.addWidget(self.label_time_domain)
        sub_layout_h_3.addWidget(self.enter_time_domain)
        sub_layout_h_4.addWidget(self.label_frequency_domain)
        sub_layout_h_4.addWidget(self.enter_frequency_domain)
        sub_layout_h_5.addWidget(self.label_out_opt)
        sub_layout_h_5.addWidget(self.enter_out_opt)
        sub_layout_h_6.addWidget(self.begin_button)
        sub_layout_h_6.addWidget(self.preview_button)
        sub_layout_v.addLayout(sub_layout_h)
        sub_layout_v.addLayout(sub_layout_h_2)
        sub_layout_v.addLayout(sub_layout_h_3)
        sub_layout_v.addLayout(sub_layout_h_4)
        sub_layout_v.addLayout(sub_layout_h_5)
        sub_layout_v.addLayout(sub_layout_h_6)
        sub_layout_v.addWidget(self.log_widget,1)
        sub_layout_v.setAlignment(Qt.AlignTop)

        main_layout.addLayout(sub_layout_v,0)
        main_layout.addWidget(self.graph_widget,1)
        self.setLayout(main_layout)

    def preview(self):
        self.controler.preview()

    def begin_optimization(self):
        global graph_option_2
        if self.controler.is_temp_file_1 == 1:
            if self.controler.is_temp_file_2 == 1:
                try:
                    self.time_domain  = str(self.enter_time_domain.text())
                    if self.time_domain=="":
                        self.controler.error_message_output_filename()
                        return(0)
                except:
                    self.controler.error_message_output_filename()
                    return(0)
                try:
                    self.frequency_domain = str(self.enter_frequency_domain.text())
                    if self.frequency_domain=="":
                        self.controler.error_message_output_filename()
                        return(0)
                except:
                    self.controler.error_message_output_filename()
                    return(0)
                try:
                    self.out_opt_filename  = str(self.enter_out_opt.text())
                    if self.out_opt_filename=="":
                        self.controler.error_message_output_filename()
                        return(0)
                except:
                    self.controler.error_message_output_filename()
                    return(0)
                try:
                    self.controler.get_output_paths(self.outputdir,self.time_domain,self.frequency_domain, self.out_opt_filename)
                except:
                    self.controler.error_message_output_paths()
                    return(0)
                try:
                    from mpi4py import MPI
                    nb_proc=int(self.enter_nb_proc.text())
                except:
                     self.controler.message_log_tab3("You don't have MPI for parallelization, we'll use only 1 process")
                     nb_proc=1
                self.controler.begin_optimization(nb_proc)
                graph_option_2='Real(refractive index)'
            else:
                self.controler.no_temp_file_2()
        else:
            self.controler.no_temp_file_1()


    def get_outputdir(self):
        DirectoryName = QFileDialog.getExistingDirectory(self,"Select Directory")
        try:
            self.outputdir=str(DirectoryName)
            name=self.controler.name_file(str(DirectoryName))
            self.button_outputdir.setText(name)
        except:
            self.controler.error_message_path3()

    def pressed_loading(self):
        self.controler.loading_text3()


class log_optimisation(QTextEdit):
    def __init__(self, parent, controler):
        super().__init__(parent)
        self.controler = controler
        self.controler.addClient3(self)
#        self.setFixedHeight(150)
        self.setReadOnly(True)
        self.append("Optimisation")

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


        # Organisation layout
        self.vlayoutmain = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.button_real_index)
        self.hlayout.addWidget(self.button_im_index)
        self.hlayout.addWidget(self.button_E_field)
        self.hlayout.addWidget(self.button_E_field_dB)
        self.hlayout.addWidget(self.button_Pulse_E_field)
        self.hlayout.addWidget(self.button_Pulse_E_field_dB)
        self.vlayoutmain.addWidget(self.canvas)
        self.vlayoutmain.addLayout(self.hlayout)
        self.setLayout(self.vlayoutmain)
#        self.drawgraph()


    def draw_graph_init(self,myinputdata,myinputdatafromfile,myfitteddata,
                        monepsilon,myglobalparameters):
        global graph_option_2
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        if graph_option_2=='Real(refractive index)':
            ax1.set_title('Real part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Real part of refractive index',color=color)
            ax1.plot(myglobalparameters.freq, np.sqrt(monepsilon).real, 'b-', label='target')
            ax1.plot(myglobalparameters.freq, np.sqrt(myfitteddata.epsilon).real, 'g-', label='fited')
            ax1.legend()

#            ax1.tick_params(axis='y', labelcolor=color)

        elif graph_option_2=='Im(refractive index)':
            ax1.set_title('Imaginary part of refractive index', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Imaginary part of refractive index',color=color)
            ax1.plot(myglobalparameters.freq, np.sqrt(monepsilon).imag, 'r-', label='target')
            ax1.plot(myglobalparameters.freq, -np.sqrt(myfitteddata.epsilon).imag, 'g-', label='fited')
            ax1.legend()

        elif graph_option_2=='E_field':
            ax1.set_title('E_field', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field',color=color)
            ax1.plot(myglobalparameters.freq,(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='reference spectre (log)')
            ax1.plot(myglobalparameters.freq,(abs(myinputdata.Spulse))/np.log(10), 'b-', label='spectre after sample (log)')
            ax1.plot(myglobalparameters.freq,(abs(myfitteddata.Spulse))/np.log(10), 'r-', label='reference spectre (log)')
            ax1.legend()

        elif graph_option_2=='E_field [dB]':
            ax1.set_title('E_field [dB]', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('E_field [dB]',color=color)
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='reference spectre (log)')
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myinputdata.Spulse))/np.log(10), 'b-', label='spectre after (log)')
            ax1.plot(myglobalparameters.freq,20*np.log(abs(myfitteddata.Spulse))/np.log(10), 'r-', label='fited spectre (log)')

            ax1.legend()

        elif graph_option_2=='Pulse (E_field)':
            ax1.set_title('Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field)',color=color)
            ax1.plot(myglobalparameters.t, myinputdatafromfile.Pulseinit, 'g-', label='reference pulse')
            ax1.plot(myglobalparameters.t, myinputdata.pulse, 'b-', label='pulse after sample')
            ax1.plot(myglobalparameters.t, myfitteddata.pulse, 'r-', label='fited pulse')
            ax1.legend()

        else:
            ax1.set_title('Pulse (E_field)', fontsize=10)
            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Pulse (E_field)',color=color)
            ax1.plot(myglobalparameters.t, 10*np.log(myinputdatafromfile.Pulseinit**2)/np.log(10), 'g-', label='reference pulse')
            ax1.plot(myglobalparameters.t, 10*np.log(myinputdata.pulse**2)/np.log(10), 'b-', label='pulse after sample')
            ax1.plot(myglobalparameters.t, 10*np.log(myfitteddata.pulse**2)/np.log(10), 'r-', label='pulse fited')
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

    def refresh(self):
        try:
            try:
                monepsilon=self.controler.monepsilon
                myinputdata=self.controler.myinputdata
                myinputdatafromfile=self.controler.myinputdatafromfile
                myfitteddata=self.controler.myfitteddata
                myglobalparameters=self.controler.myglobalparameters
                self.draw_graph_init(myinputdata,myinputdatafromfile,myfitteddata,
                                     monepsilon,myglobalparameters)
            except:
                monepsilon=self.controler.monepsilon
                myinputdata=self.controler.myinputdata
                myinputdatafromfile=self.controler.myinputdatafromfile
                myfitteddata=self.controler.previewdata
                myglobalparameters=self.controler.myglobalparameters
                self.draw_graph_init(myinputdata,myinputdatafromfile,myfitteddata,
                                     monepsilon,myglobalparameters)
        except:
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
            pass

def main():
    app = QApplication([])
    controler = Controler()
    win = MainWindow(controler)
    controler.init()
    win.show()
    app.exec()

if __name__ == '__main__':
    main()
