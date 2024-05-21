# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math, cmath

# from __future__ import print_function
from pyswarm import pso
import random
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from numpy import *
from pylab import *
from scipy.io.matlab import mio

# from scipy import optimize
from scipy.optimize import minimize

###############################################################################
###############################################################################
j = 1j
c = 2.998e8


###############################################################################

# defining the Error function
def errorFct(n, k, d, frequency, measAmp, measAngle, FP):
    """ Function to create a model giving, the frequencies, its "n", its "k", the thickness of the sample in m
    and if the Fabry-Perot effect is taken into account"""
    omeg = 2 * pi * frequency
    nair = 1  # Refractive index of Air
    refInd = n - 1j * k
    # Fresnel coefficients (transmission)
    Taf = 2 * nair / (nair + (refInd))  # Air => Sample
    Tfa = 2 * (refInd) / (nair + (refInd))  # Sample => Air
    Tpe = exp(-1j * (refInd - nair) * omeg * d / c)  # Propagation

    # FP_effect is the term taking into account Fabry-Perot multiple reflections in the sample
    FP_effect = 1

    if FP == 0:
        FP_effect = 1 / (
            1
            - ((refInd - nair) / (refInd + nair)) ** 2
            * exp(-2 * 1j * refInd * omeg * d / c)
        )
        z = Taf * Tfa * Tpe * FP_effect
    else:
        z = Taf * Tfa * Tpe 

    angleTaf = angle(Taf)
    angleTfa = angle(Tfa)
    angleTfe = -(n - nair) * omeg * d / c
    angleFP = angle(FP_effect)

    #à utiliser en premier lieu
    #ThZ = abs(z)*abs(exp(j*(angleTaf+angleTfa+angleTfe+angleFP)))
    #measZ = measAmp*abs(exp(j*measAngle))
    #erreur = abs((ThZ-measZ)**2)

    #à utiliser si la version en exp ne marche pas (partie réelle de l'indice de réfraction inversée)
    ThZ = (log(abs(z))+j*(angleTaf+angleTfa+angleTfe+angleFP))
    measZ = ((log(measAmp))+j*measAngle)
    erreur = abs((ThZ-measZ)**2)

    #version antérieure, pas bon car séparait la formule en deux
    #errorAngle = (angleTaf + angleTfa + angleTfe + angleFP - measAngle) ** 2
    #errorAbs = (log(abs(z)) - log(measAmp)) ** 2
    #erreur = abs(errorAngle + errorAbs)

    return erreur


# defining the 'so-called' inverse problem to get the refractive index
def inverseProblem(freq, fctTrans, thick, FP, init, limitUp):
    """ Calculates n and k over frequency.
        freq : Frequency in Hz, must be POSITIVE;
        fctTrans : Transfer function, must be same size as freq and correspond to positive frequencies;
        thick : thickness of the sample in meters [m];
        FP : If equal to 1 takes into account the Farby-Perot effect;
        init : initial guess for the inverse problem. init should be an array : init = [3, 0.1], for example;
        limitUp : High frequency limit of the study in Hz;
    """
    init = np.transpose([init.real, init.imag])
    f = freq[(freq <= limitUp)]

    ## The measurement
    meas = fctTrans[(freq <= limitUp)]

    ## Inverse problem
    nInv = zeros(freq.shape[0])
    kInv = zeros(freq.shape[0])

    measAngleTh = unwrap(angle(meas))
    measAbsTh = abs(meas)
    for i in range(0, f.shape[0] - 1):
        pouet = lambda x: errorFct(
            x[0], x[1], thick, f[i], measAbsTh[i], measAngleTh[i], FP
        )
        res = minimize(pouet, init[i], method="Nelder-Mead", tol=1e-6)
        # cf https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
        # for other methods
        nInv[i] = res.x[0]
        kInv[i] = res.x[1]

    ref_index = j * kInv + nInv

    return ref_index

def dielcal(mytransferfunction, z, myglobalparameters,FP ,scattering=None):
    global zz, mytransferfunction2


    """ 
    Takes variables from the the input and uses the inverseProblem
    The first part is the initialization and the last line is the optimisation
    """
    mytransferfunction2 = np.copy(mytransferfunction)
    zz = np.copy(z)
    ref_index = np.ones(len(myglobalparameters.w))
    myangleTF = np.unwrap(np.angle(mytransferfunction))

    for compt in range(
        0, 3
    ):  # the max of this loop was choosen to optimize the effective index for a 250micron thick quartz substrate it gives good result at 1/1000 but shows weird convergeance (larger is not better so this may had to be changed with the sample)
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w * (ref_index) / c * zz)
        myangleTrans = np.nan_to_num(np.angle(tt))  # ici il y a un truc a bien regarder
        phase = np.unwrap(
            myangleTF + myangleTrans
        )  # we are taking the phase from the transfer function
        
        naprox = 1 + abs(
            phase * c / zz / myglobalparameters.w
        )  # this gives the real part of the index
        
        ref_index = naprox + j * (ref_index.imag)  # this gives the real part of the index
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w * (ref_index) / c * zz)
        kaprox = -np.nan_to_num(
            c / z / myglobalparameters.w * np.log(abs(mytransferfunction) / (tt))
        )
        ref_index = j * kaprox.real + ref_index.real
    ref_index[0] = 1

    ref_index2 = inverseProblem(
        myglobalparameters.w / 2 / np.pi, mytransferfunction2, zz, FP, ref_index, 6e12
    )

    return np.array(ref_index2 ** 2)
