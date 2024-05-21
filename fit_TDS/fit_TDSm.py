#!/usr/bin/python
# -*- coding: latin-1 -*-


import numpy as np   ## Library to simplify the linear algebra calculations



j = 1j
c = 2.998e8
h = 6.62607015E-34
k= 1.38064852E-23



class Model:
    def __init__(self):
        self.name = ""
        self.label = ""                          #label used to ask for the number of terms of the given model
        self.explanation = ""                    #Text displayed in the reference box
        self.isCumulative = False                #Can there be several terms with this model?
        self.variableNames = []                  
        self.variableUnits =  []
        self.variableDescriptions = []           #if isCumulative=True, the index of the term will be added to the descriptions (\n is added anyway)
        self.invalidNumberMessage = ""           #Message to show if the number of term entered is invalid

    def epsilon(self,eps,w,paramList):
        raise NotImplementedError()              #raises an exception if you use a model without implementing its epsilon method

    def gauss(self,w,sigma):
        return np.exp(-((w))**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    def lor(self,w,chi,w0,gamma):
        return (chi*w0)/(w0**2+j*gamma*w-w**2) #normalis� par w0 car int�gr�

    def debye(self,w,chi,tau): 
        return chi/(1+j*w*tau)

    def lnl(self,nu,sigma):
        return np.exp(-(np.log(nu))**2/(2*sigma**2))*1/(np.sqrt(2*np.pi)*sigma*nu)

    def boltzmann(self,nu,nu0,T):
        return (1-np.exp(-(nu*h)/(k*T))-((nu)*(h/(k*T)))*np.exp(-(nu)*(h/(k*T)))-1/2*((nu)*(h/(k*T)))**2*np.exp(-(nu)*(h/(k*T))))*np.heaviside(nu-nu0, 1)
####################################
######### Material Models ##########
####################################

class Drude(Model):
    def __init__(self):
        self.name = "Drude"
        self.label = "Drude term"
        self.explanation = "Drude model depicts the permitivity Epsillon as Eps =Eps_0- Omega_p^2/(Omega^2-j*gamma*omega)."
        self.isCumulative = False
        self.variableNames = ["Omega_p","gamma"]
        self.variableUnits =  ["radian/s","radian/s"]
        self.variableDescriptions = ["Drude's Model Plasma frequency",
                                      "Drude damping rate"]
        self.invalidNumberMessage = ""


    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        wp = variableDictionary.get("Omega_p")
        gamma = variableDictionary.get("gamma")
        return -wp**2/(1E0+w**2-j*gamma*w)

class Popov(Model):
    def __init__(self):
        self.name = "Popov"
        self.label = "Popov term"
        self.explanation = "Popov model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon/[1+[[j*Omega*taudef]^-1+[j*Omega*tauosc]^-delta]^-1]]."
        self.isCumulative = False
        self.variableNames = ['Delta_Epsilon_Popov', 'Tau_{def}', 'Tau_{osc}', 'delta']     
        self.variableUnits =  ["dimensionless", "s", "s", "dimensionless"]
        self.variableDescriptions = ["Oscillator strength of the mode", 'Time constant of defects', 'Time constant of oscillations', 'delta']
        self.invalidNumberMessage = ""

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get("Delta_Epsilon_Popov")
        taudef = variableDictionary.get("Tau_{def}")
        tauosc = variableDictionary.get("Tau_{osc}")
        delta = variableDictionary.get("delta")
        return chi/(1E0+1/(np.exp(-np.log(j*w*taudef+j*1e-30))+np.exp(-delta*np.log(j*w*tauosc+j*1e-30))))
        #return chi/(1E0+((1e-10*1j+j*w*taudef)**(-1)+(1j*1e-10+j*w*tauosc)**(-delta))**(-1))

class Scattering(Model):
    def __init__(self):
        self.name = "Scattering"
        self.label = "Scattering"
        self.explanation = "Do you want to take into account scattering ?"
        self.isCumulative = False
        self.variableNames = ["Beta","Scat_freq_min","Scat_freq_max"]
        self.variableUnits =  ["1/m","Hz","Hz"]
        self.variableDescriptions = ["Loss coefficient",
                                      "Beginning frequency of scattering",
                                      "Ending frequency of scattering"]
        self.invalidNumberMessage = ""

    def epsilon(self,eps,w,paramList):        
        variableDictionary = dict(zip(self.variableNames, paramList))
        beta = variableDictionary.get("Beta")
        Scat_freq_min = variableDictionary.get("Scat_freq_min")
        Scat_freq_max = variableDictionary.get("Scat_freq_max")
        omega = np.where(w<Scat_freq_max*2*np.pi,w,1e-299)
        omega = np.where(omega>Scat_freq_min*2*np.pi,omega,1e-299)
        alpha = beta*(omega/(2*np.pi*1e12))**3
        n_diff = - j*alpha
        return n_diff**2+2*np.sqrt(eps)*n_diff

class Lorentz(Model):
    def __init__(self):
        self.name = "Lorentz"
        self.label = "Number of Lorentz oscillators"
        self.explanation = "Lorentz model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon*Omega_0^2]/[Omega_0^2+j*gamma*Omega-Omega^2]."
        self.isCumulative = True
        self.variableNames = ['Delta_Epsillon_Lorentz',
                               '1/(2pi)*Omega0_Lorentz',
                               '1/(2pi)*Gamma']
        self.variableUnits =  ["dimensionless", "Hz","Hz"]
        self.variableDescriptions = ["Oscillator strentgh of the mode #",
                                      'Frequency of the mode #',
                                      'Linewidth of the mode #']
        self.invalidNumberMessage = "Invalid number of Lorentz Oscillators."

    def epsilon(self,eps,w,paramList):        
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get('Delta_Epsillon_Lorentz')
        w0 = variableDictionary.get('1/(2pi)*Omega0_Lorentz')*2*np.pi
        gamma = variableDictionary.get('1/(2pi)*Gamma')*2*np.pi
        return chi*w0**2/(w0**2+j*gamma*w-w**2)

class Voigt(Model):
    def __init__(self):
        self.name = "Voigt"
        self.label = "Number of Voigt oscillators"
        self.explanation = "Voigt profile depicts the permitivity Epsillon as Eps = Eps_0 + Gaussian * Lorentzian."
        self.isCumulative = True
        self.variableNames = ["sigma","Chi","Nu_0","GammaVoigt"]
        self.variableUnits =  ["dimensionless","dimensionless","Hz","Hz"]
        self.variableDescriptions = ['Width (sigma of the Gaussian) of the mode #',
                                      "Strength of the Lorentzian", 'Central frequency of the mode #', 
                                      'Width of the Lorentzian of the mode #'] 
        self.invalidNumberMessage = "Invalid number of Voigt Oscillators."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        wneg = -w[::-1]
        wtot=np.append(wneg,w)
        dw=w[2]-w[1]
        chi = variableDictionary.get('Chi')
        gamma = variableDictionary.get('GammaVoigt')*2*np.pi
        sigma = variableDictionary.get('sigma')*2*np.pi
        w0 = variableDictionary.get('Nu_0')*2*np.pi
        gauss00 = self.gauss(0,sigma)
        somme0 = self.lor(wtot,chi,w0,gamma)*gauss00
        norm0 = gauss00/w0
        if gamma/10 >= dw:
            step0 = (gamma/(2*np.pi))/10
        else:
            step0 = dw/(2*np.pi)
        count0 = step0            
        gauss0 = gauss00
        while gauss0 > gauss00/1000:
            gauss0 = self.gauss((count0*2*np.pi+w0)-w0,sigma)
            somme0 = somme0 + (self.lor(wtot,chi,(count0*2*np.pi+w0),gamma)+self.lor(wtot,chi,(-count0*2*np.pi+w0),gamma))*gauss0
            norm0 = norm0 + 2*gauss0/((count0*2*np.pi+w0))
            count0 = count0 + step0
        C0 = somme0/norm0
        return C0[len(wtot)-len(w):len(wtot)]
        
class LogN(Model):
    def __init__(self):
        self.name = "LogN"
        self.label = "Number of log-normal distributions"
        self.explanation = "Log-Normal distribution depicts the permitivity Epsillon as Eps = Eps_0 + Log-Normal * Debye."
        self.isCumulative = True
        self.variableNames = ["sigma_LN","Chi_LN","tau_LN"]
        self.variableUnits =  ["Hz","dimensionless","s"]
        self.variableDescriptions = ['Width of the Log-normal of the mode #','Oscillator strentgh of the mode #','Time constant of the mode #']

        self.invalidNumberMessage = "Invalid number of Log-normal distributions."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        dw=w[2]-w[1]
        wneg = -w[::-1]
        wtot=np.append(wneg,w)
        chi = variableDictionary.get('Chi_LN')
        sigma = variableDictionary.get('sigma_LN')*2*np.pi
        tau = variableDictionary.get('tau_LN')
        nu0=1/(2*np.pi*tau)
        
        lnl11 = self.lnl(nu0,0,sigma)
        somme1 = self.debye(wtot,chi,tau)*lnl11
        norm1 = lnl11
        step1 = dw/(2*np.pi)
        count1 = step1
        lnl1 = lnl11
        while lnl1 > lnl11/1000:
            lnl1 = self.lnl((count1+nu0),0,sigma)
            somme1 = somme1 + (self.debye(wtot,chi,1/((count1+nu0)*2*np.pi)))*lnl1#+debye(wtot,chi,1/((-count1+nu0)*2*np.pi)))*lnl1
            norm1 = norm1 + 2*lnl1
            count1 = count1 + step1
        C1 = somme1/norm1
        return C1[len(wtot)-len(w):len(wtot)]

class Debye(Model):
    def __init__(self):
        self.name = "Debye"
        self.label = "Number of Debye oscillators"
        self.explanation = "Debye model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon/[1+j*Omega*tau]."
        self.isCumulative = True
        self.variableNames = ["Delta_Epsillon_Debye","tau"]
        self.variableUnits =  ["dimensionless", "s"]     
        self.variableDescriptions = ["Oscillator strentgh of the mode #", 'Time constant #']

        self.invalidNumberMessage = "Invalid number of Debye Oscillators."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get('Delta_Epsillon_Debye')
        tau = variableDictionary.get('tau')
        return chi/(1+j*w*tau)


class HN(Model):
    def __init__(self):
        self.name = "HN"
        self.label = "Number of Havriliak-Negami terms"
        self.explanation = "Havriliak-Negami model depicts the permitivity Epsilon as Eps = Eps_0 + [ Delta_epsillon/[1+(j*Omega*tau)^alpha]^beta."
        self.isCumulative = True
        self.variableNames = ["Delta_Epsillon_HN","tau_HN","Alpha","Beta_HN"]
        self.variableUnits =  ["dimensionless", "s", "dimensionless", "dimensionless"]      
        self.variableDescriptions = ["Oscillator strentgh of the mode #", 'Time constant #', "Alpha #", "Beta #"]
        self.invalidNumberMessage = "Invalid number of Havriliak-Negami terms."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get('Delta_Epsillon_HN')
        tau = variableDictionary.get('tau_HN')
        alpha = (variableDictionary.get('Alpha'))
        beta = (variableDictionary.get('Beta_HN'))
        eps = chi/(1+(j*w*tau)**alpha)**beta
        return eps

class Boltzmann(Model):
    def __init__(self):
        self.name = "Boltzmann_continuum"
        self.label = "Number of Boltzmann terms"
        self.explanation = "Boltzmann continuum model depicts the permitivity Epsilon as a convolution of a Boltzmann distribution and a Lorentz term."
        self.isCumulative = True
        self.variableNames = ["Chi_Boltzmann","nu0_Boltzmann","Gamma_Boltzmann","Temp_Boltzmann"]
        self.variableUnits =  ["dimensionless", "Hz", "Hz", "K"]      
        self.variableDescriptions = ["Oscillator strentgh of the mode #", 'Frequency of the mode #', 'Linewidth of the mode #', "Temperature of the mode #"]
        self.invalidNumberMessage = "Invalid number of Boltzmann continuum terms."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get('Chi_Boltzmann')
        nu0 = variableDictionary.get('nu0_Boltzmann') #2pi ?
        gamma = variableDictionary.get('Gamma_Boltzmann')*2*np.pi
        T = variableDictionary.get('Temp_Boltzmann')
        wneg = -w[::-1]
        wtot=np.append(wneg,w)
        dw = w[2]-w[1]
        nutot=wtot/(2*np.pi)
        somme = np.zeros(len(nutot))
        norm = np.zeros(len(nutot))
        if gamma/10 >= dw:
            step = (gamma/(2*np.pi))/10
        else:
            step = dw/(2*np.pi)
        count = nu0
        while count < nutot[-1]:
            somme = somme + self.lor(wtot,chi,count*2*np.pi,gamma)*self.boltzmann(count,nu0,T)
            norm = norm + self.boltzmann(count,nu0,T)/(count*2*np.pi)
            count = count + step
        C2 = somme/norm
        eps = C2[len(wtot)-len(w):len(wtot)]
#        chi = variableDictionary.get('Chi_Boltzmann')
#        nu0 = variableDictionary.get('nu0_Boltzmann')
#        gamma = np.log(variableDictionary.get('Gamma_Boltzmann'))
#        T = np.log(variableDictionary.get('Temp_Boltzmann'))
#        wneg = -w[::-1]
#        wtot=np.append(wneg,w)
#        dw = w[2]-w[1]
#        nutot=wtot/(2*np.pi)
#        somme = np.zeros(len(nutot))
#        norm = np.zeros(len(nutot))
#        if gamma/10 >= dw:
#            step = (gamma/(2*np.pi))/10
#        else:
#            step = dw/(2*np.pi)
#        count = nu0
#        while count < nutot[-1]:
#            somme = somme + self.lor(wtot,chi,count*2*np.pi,gamma)/(count*2*np.pi)*self.boltzmann(count,nu0,T)
#            norm = norm + self.boltzmann(count,nu0,T)/(count*2*np.pi)
#            count = count + step
#        C2 = somme/norm
#        return C2[len(wtot)-len(w):len(wtot)]
        return eps
    
class Roccard(Model):
    def __init__(self):
        self.name = "Roccard"
        self.label = "Number of Roccard terms"
        self.explanation = "Roccard model depicts the permitivity Epsillon as Eps = Eps_0 +[ Delta_epsillon/[(1+j*Omega*tau1).(1+j*Omega*tau2)]."
        self.isCumulative = True
        self.variableNames = ["Delta_Epsillon_Roccard","tau_Rocc1","tau_Rocc2"]
        self.variableUnits =  ["dimensionless", "s", "s"]     
        self.variableDescriptions = ["Oscillator strength of the mode #", 'Time constant #', 'Friction time #']

        self.invalidNumberMessage = "Invalid number of Roccard terms."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        chi = variableDictionary.get('Delta_Epsillon_Roccard')
        tau1 = variableDictionary.get('tau_Rocc1')
        tau2 = variableDictionary.get('tau_Rocc2')
        return chi/((1+j*w*tau1)*(1+j*w*tau2))
    
class Titov33(Model):
    def __init__(self):
        self.name = "Titov33"
        self.label = "Number of Titov terms"
        self.explanation = "Titov model depicts the permitivity Epsillon as Eps = Eps_0 +."
        self.isCumulative = True
        self.variableNames = ["temperature_exp","Beta1","Beta2","Gamma1","Eta","SigmaV"]
        self.variableUnits =  ["Celsius", "dimensionless", "dimensionless","dimensionless", "s","dimensionless"]     
        self.variableDescriptions = ["temperature_exp", "Beta1","Beta2","Gamma1","Eta","SigmaV"]

        self.invalidNumberMessage = "Invalid number of Titov terms."

    def epsilon(self,eps,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        
        T = variableDictionary.get('temperature_exp')
        beta1 = variableDictionary.get('Beta1')
        beta2 = variableDictionary.get('Beta2')
        gamma1 = variableDictionary.get('Gamma1')
        eta = variableDictionary.get('Eta')
        sigmaV = variableDictionary.get('SigmaV')
        #tauD=variableDictionary.get('TauD')
        teta = (1-300/(273.15+T))
        eps0 = 77.66 - 103.3*teta
        eps_inf = eps[0]
        tauBF = 1/(2*np.pi*(20.27+146.5*teta+314*teta**2))*1e-9
        tauD = tauBF / (eta * (1/(2*beta2**2)-1/sigmaV+1))
        deltaeps = eps0 - eps_inf
        return (deltaeps/(1+j*eta*w*tauD))*(((2*beta2**2)/(2*beta2**2+j*eta*w*tauD))+((j*eta*w*tauD)/(sigmaV-2*(eta*w/gamma1)**2+2*j*beta1*eta*w/gamma1)))
############################################################
############################################################
materialModels = [Drude(),Popov(),Lorentz(),Voigt(),LogN(),Debye(),HN(),Boltzmann(),Roccard(),Titov33(),Scattering()] #list of the models that will be used in the software

#NOTE: Scattering() must be last in this list, as its epsilon function depends on the previous value of epsilon


####################################
######### Interface Models #########
####################################



class InterfaceModel:
    def __init__(self):
        self.name = ""
        self.label = ""                          #label used to ask for the number of terms of the given model
        self.explanation = ""                    #Text displayed in the reference box
 #       self.isCumulative = False                #Can there be several terms with this model?
        self.variableNames = []                  
        self.variableUnits =  []
        self.variableDescriptions = []           #if isCumulative=True, the index of the term will be added to the descriptions (\n is added anyway)
 #       self.invalidNumberMessage = ""           #Message to show if the number of term entered is invalid

    def H(self,w,paramList):
        raise NotImplementedError()              #raises an exception if you use a model without implementing its transmission/reflection model


class TDCMT(InterfaceModel):
    def __init__(self):
        self.name="TDCMT"
        self.label="TDCMT"
        self.explanation="Time Domain Coupled Mode Theory"
        self.variableNames=["f0","dec0","dece"]
        self.variableUnits=["Hz","s^-1","s^-1"]
        self.variableDescriptions=["Central frequency of the mode of the resonator #",
                                   "Non radiative decay rate of the mode #",
                                   "Radiative decay rate of the mode of the resonator #"]
        
    def H(self,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        w0=variableDictionary.get('f0')*2*np.pi
        dec0=variableDictionary.get('dec0') #decay rate = 1/tau
        dece=variableDictionary.get('dece')
        return dece/((j*(w-w0)+dece+dec0))

class TDCMT2(InterfaceModel):
    def __init__(self):
        self.name="TDCMT2"
        self.label="2nd order oscillator"
        self.explanation="Similar to Time Domain Coupled Mode Theory, but without neglecting the negative frequencies"
        self.variableNames=["f0","dec0","dece"]
        self.variableUnits=["Hz","s^-1","s^-1"]
        self.variableDescriptions=["Central frequency of the mode of the resonator #",
                                   "Non radiative decay rate of the mode #",
                                   "Radiative decay rate of the mode of the resonator #"]
        
    def H(self,w,paramList):
        variableDictionary = dict(zip(self.variableNames, paramList))
        w0=variableDictionary.get('f0')*2*np.pi
        dec0=variableDictionary.get('dec0') #decay rate = 1/tau
        dece=variableDictionary.get('dece')
        return dece*2*j*w/((j*(w-w0)+dece+dec0)*(j*(w+w0)+dece+dec0))

interfaceModels=[TDCMT(),TDCMT2()]