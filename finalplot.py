import os, sys, time, math
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

j = 1j
##############################################################################
##############################################################################
def plotinput(monepsilon,myinputdata,myinputdatafromfile,z,myglobalparameters):
	##############################################################################
	##############################################################################
	plt.figure("Input", figsize=(12, 8))
	##############################################################################
	##############################################################################
	plt.subplot(221)
	#dans en frequence affiche epsilon
	plt.plot(myglobalparameters.freq, abs(myinputdata.Spulse/myinputdatafromfile.Spulseinit), 'b-', label='transmission absolut')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Transmission')
	plt.legend()

	plt.subplot(222)
	plt.plot(myglobalparameters.freq,  np.real((myinputdata.epsilon)), 'b-', label='real part') 
	plt.plot(myglobalparameters.freq, np.imag((myinputdata.epsilon)), 'r-', label='imaginary part')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Permitivity')
	plt.legend()

	plt.subplot(223)
	plt.plot(myglobalparameters.freq, np.log(abs(myinputdata.Spulse/myinputdatafromfile.Spulseinit))/z, 'b-', label='transmission [1/m]')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Losses [1/m]')
	plt.legend()

	plt.subplot(224)
	plt.plot(myglobalparameters.freq, 20*np.log(abs(myinputdata.Spulse))/np.log(10), 'b-', label='with sample')
	plt.plot(myglobalparameters.freq, 20*np.log(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='initial pulse')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Spectral density of energy [dB]')
	plt.legend()
	##############################################################################
	##############################################################################

	plt.figure("Time trace", figsize=(12, 5))
	##############################################################################
	##############################################################################

	#en temps affiche l'errer de calcul
	plt.plot(myglobalparameters.t, myinputdata.pulse, 'r-', label='with sample')
	plt.plot(myglobalparameters.t, myinputdatafromfile.Pulseinit, 'g-', label='initial pulse')
	plt.xlabel('Time [s]')
	plt.ylabel('E_field [a.u.]')
	plt.legend()
	plt.show()

##############################################################################
##############################################################################

def plotall(myinputdata,myinputdatafromfile,myfitteddata,monepsilon,myglobalparameters):
	
	##############################################################################
	##############################################################################
	plt.figure("Comparison of the refractive indices", figsize=(12, 8))
	##############################################################################
	##############################################################################

	plt.subplot(221)
	plt.plot(myglobalparameters.freq, np.sqrt(monepsilon).real, 'b-', label='target')
	plt.plot(myglobalparameters.freq, np.sqrt(myfitteddata.epsilon).real, 'g-', label='fited')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Real part of refractive index')
	plt.legend()

	plt.subplot(222)
	plt.plot(myglobalparameters.freq, np.sqrt(monepsilon).imag, 'r-', label='target')
	plt.plot(myglobalparameters.freq, -np.sqrt(myfitteddata.epsilon).imag, 'g-', label='fited')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Imaginary part of refractive index')
	plt.legend()

	plt.subplot(223)
	plt.plot(myglobalparameters.freq, np.sqrt(myfitteddata.epsilon).real-np.sqrt(monepsilon).real, 'b-', label='in the real part of refractive index')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Discrepency')
	plt.legend()

	plt.subplot(224)
	plt.plot(myglobalparameters.freq, np.sqrt(myfitteddata.epsilon).imag-np.sqrt(monepsilon).imag, 'r-', label='in the imaginary part of refractive index')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Discrepency')
	plt.legend()

	##############################################################################
	##############################################################################
	plt.figure("Comparison of spectra", figsize=(12, 8))
	##############################################################################
	##############################################################################
	plt.subplot(221)
	plt.plot( myglobalparameters.freq,(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='spectre initial log')
	plt.plot( myglobalparameters.freq,(abs(myinputdata.Spulse))/np.log(10), 'b-', label='spectre after log')
	plt.plot( myglobalparameters.freq,(abs(myfitteddata.Spulse))/np.log(10), 'r-', label='spectre fited log')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('E_field ')
	plt.legend()

	plt.subplot(223)
	plt.plot( myglobalparameters.freq,20*np.log(abs(myinputdatafromfile.Spulseinit))/np.log(10), 'g-', label='spectre initial log')
	plt.plot( myglobalparameters.freq,20*np.log(abs(myinputdata.Spulse))/np.log(10), 'b-', label='spectre after log')
	plt.plot( myglobalparameters.freq,20*np.log(abs(myfitteddata.Spulse))/np.log(10), 'r-', label='spectre fited log')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('E_field [dB]')
	plt.legend()

	plt.subplot(222)
	#l'erreur su le spectre abs et angle
	plt.plot( myglobalparameters.freq,abs(myfitteddata.Spulse-myinputdata.Spulse), 'r-', label='spectre fited log')
	plt.plot( myglobalparameters.freq,abs(np.angle(myfitteddata.Spulse)-np.angle(myinputdata.Spulse)), 'b-', label='angle fited')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('delta E_field / delta Phase')
	plt.legend()

	plt.subplot(224)
	#l'erreur su le spectre abs et angle db
	plt.plot( myglobalparameters.freq,20*np.log(abs(myfitteddata.Spulse-myinputdata.Spulse))/np.log(10), 'r-', label='spectre fited log')
	plt.plot( myglobalparameters.freq,20*np.log(abs(np.angle(myfitteddata.Spulse)-np.angle(myinputdata.Spulse)))/np.log(10), 'b-', label='angle fited')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('delta E_field / delta Phase both in dB')
	plt.legend()

	##############################################################################
	##############################################################################
	plt.figure("Comparison of impulsions", figsize=(12, 8))
	##############################################################################
	##############################################################################

	plt.subplot(221)
	#en temps affiche l'impulsion supperpose a l'initiale
	plt.plot(myglobalparameters.t, myinputdata.pulse, 'b-', label='pulse data')
	plt.plot(myglobalparameters.t, myinputdatafromfile.Pulseinit, 'g-', label='pulse init')
	plt.plot(myglobalparameters.t, myfitteddata.pulse, 'r-', label='pulse fited')
	plt.xlabel('Time [s]')
	plt.ylabel('Pulse (E_field)')
	plt.legend()

	plt.subplot(223)
	#en temps affiche l'errer de calcul
	plt.plot(myglobalparameters.t, myinputdata.pulse-myfitteddata.pulse, 'b-', label='Error')
	plt.xlabel('Time [s]')
	plt.ylabel('Error (E_field)')
	plt.legend()
    
	plt.subplot(222)
	#en temps affiche l'impulsion supperpose a l'initiale
	plt.plot(myglobalparameters.t, 10*np.log(myinputdata.pulse**2)/np.log(10), 'b-', label='pulse data')
	plt.plot(myglobalparameters.t, 10*np.log(myinputdatafromfile.Pulseinit**2)/np.log(10), 'g-', label='pulse init')
	plt.plot(myglobalparameters.t, 10*np.log(myfitteddata.pulse**2)/np.log(10), 'r-', label='pulse fited')
	plt.xlabel('Time [s]')
	plt.ylabel('Pulse (E_field)')
	plt.legend()

	plt.subplot(224)
	#en temps affiche l'errer de calcul
	plt.plot(myglobalparameters.t, 20*np.log(abs(myinputdata.pulse-myfitteddata.pulse))/np.log(10), 'b-', label='Error')
	plt.xlabel('Time [s]')
	plt.ylabel('Error [dB]')
	plt.legend()

	##############################################################################
	##############################################################################
	plt.figure("Comparison phase", figsize=(12, 8))
	##############################################################################
	##############################################################################
	plt.subplot(221)
	#dle spectre en lineaire
	plt.plot( myglobalparameters.freq,np.unwrap(np.angle(myfitteddata.Spulse)), 'r-', label='phase fited')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Phase')
	plt.legend()

	plt.subplot(223)
	#le spectre en db
	plt.plot( myglobalparameters.freq,np.unwrap(np.angle(myinputdata.Spulse)), 'b-', label='phase data')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Phase')
	plt.legend()

	plt.subplot(222)
	#la phase du spectre
	plt.plot( myglobalparameters.freq,np.unwrap(np.angle(myinputdatafromfile.Spulseinit)), 'g-', label='phase init')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Phase')
	plt.legend()

	plt.subplot(224)
	#l'erreur su le spectre abs et angle
	plt.plot(myglobalparameters.freq,np.unwrap(np.angle(myfitteddata.Spulse))-np.unwrap(np.angle(myinputdata.Spulse)), 'b-', label='difference phase fited-sample')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('delta E_field / delta Phase')
	plt.legend()

	##############################################################################
	##############################################################################
	##############################################################################
	##############################################################################
	plt.show()

##############################################################################
##############################################################################
