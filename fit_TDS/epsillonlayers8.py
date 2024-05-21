# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, math

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

    if FP == 1:
        FP_effect = 1 / (1-  ((refInd - nair) / (refInd + nair)) ** 2 * exp(-2 * 1j * refInd * omeg * d / c))
        z = Taf * Tfa * Tpe * FP_effect
    else:
        z = Taf * Tfa * Tpe * FP_effect

    angleTaf = angle(Taf)
    angleTfa = angle(Tfa)
    angleTfe = -(n - nair) * omeg * d / c
    angleFP = angle(FP_effect)

    errorAngle = (angleTaf + angleTfa + angleTfe + angleFP - measAngle) ** 2
    errorAbs = (log(abs(z)) - log(measAmp)) ** 2
    erreur = abs(errorAngle + errorAbs)

    return erreur

def errorFct2(n, k, d, frequency, meas, meassave, FP, eps_cell, thickn1, thickn2, fsave):
    """ Function to create a model giving, the frequencies, its "n", its "k", the thickness of the sample in m
    and if the Fabry-Perot effect is taken into account"""
    omeg=2*np.pi*frequency
    refindex1 = np.sqrt(eps_cell)
    refindex2 = np.sqrt(eps_cell)
    omegsave = 2*np.pi*fsave
    nair=1.0
    nairsave = np.ones(len(fsave))
    refInd = n - 1j * k

    refIndsave = np.ones(len(fsave))*refInd
    refindex1save = np.ones(len(fsave))*refindex1
    refindex2save = np.ones(len(fsave))*refindex2
    
    # Fresnel coefficients (transmission)
    t2a=2*refindex2save/(refindex2save+nairsave)
    ta1=2*nairsave/(refindex1save+nairsave)
    t1s=2*refindex1save/(refindex1save+refIndsave)
    ts2=2*refIndsave/(refindex2save+refIndsave)
    tt = t2a * ta1 * t1s * ts2
    
#    rs1=(refIndsave-refindex1save)/(refIndsave+refindex1save)
#    rs2=(refIndsave-refindex2save)/(refIndsave+refindex2save)
#    r1s=(refindex1save-refIndsave)/(refIndsave+refindex1save)
#    r2s=(refindex2save-refIndsave)/(refIndsave+refindex2save)
#    r1a=(refindex1save-nairsave)/(refindex1save+nairsave)
#    r2a=(refindex2save-nairsave)/(refindex2save+nairsave)
    rs1=(refIndsave-refindex1save)/(refIndsave+refindex1save)
    rs2=(refIndsave-refindex2save)/(refIndsave+refindex2save)
    r1s=(refindex1save-refIndsave)/(refIndsave+refindex1save)
    r2s=(refindex2save-refIndsave)/(refIndsave+refindex2save)
    r1a=(refindex1save-nairsave)/(nairsave+refindex1save)
    r2a=(refindex2save-nairsave)/(nairsave+refindex2save)
    
    den1=np.exp(j*omegsave*(refindex1save*thickn1+refIndsave*d+refindex2save*thickn2)/c)
    den2=-r1a*r1s*np.exp(j*omegsave*(refIndsave*d+refindex2save*thickn2-refindex1save*thickn1)/c)
    den3=-r1a*rs2*np.exp(j*omegsave*(refindex2save*thickn2-refIndsave*d-refindex1save*thickn1)/c)
    den4=r2a*r2a*r2s*r2s*np.exp(j*omegsave*(-refindex1save*thickn1-refindex2save*thickn2+refIndsave*d)/c)
    den5=-rs1*r2a*np.exp(j*omegsave*(refindex1save*thickn1-refindex2save*thickn2-refIndsave*d)/c)
    den6=-r2a*r2a*np.exp(j*omegsave*-1*(refindex1save*thickn1+refindex2save*thickn2+refIndsave*d)/c)
    den7=-rs1*rs1*np.exp(j*omegsave*(refindex1save*thickn1+refindex2save*thickn2-refIndsave*d)/c)
    den8=-r2a*r2s*np.exp(j*omegsave*(refindex1save*thickn1+refIndsave*d-refindex2save*thickn2)/c)
		
    Zsample2=tt/(den1+den2+den3+den4+den5+den6+den7+den8)
    #Zsample2=tt/(den1+den2)
    #Zref2=np.exp(-j*omegsave*nairsave*(thickn1+d+thickn2)/c)
    t1a=2*refindex1save/(refindex1save+nairsave)
    ta2=2*nairsave/(refindex2save+nairsave)
    
    ttref=t2a*ta1*t1a*ta2
    
    ra1=-r1a
    ra2=-r2a
    
    den1ref=np.exp(j*omegsave*(refindex1save*thickn1+nairsave*d+refindex2save*thickn2)/c)
    den2ref=-r1a*r1a*np.exp(j*omegsave*(nairsave*d+refindex2save*thickn2-refindex1save*thickn1)/c)
    den3ref=-r1a*ra2*np.exp(j*omegsave*(refindex2save*thickn2-nairsave*d-refindex1save*thickn1)/c)
    den4ref=r2a*r2a*r2a*r2a*np.exp(j*omegsave*(-refindex1save*thickn1-refindex2save*thickn2+nairsave*d)/c)
    den5ref=-ra1*r2a*np.exp(j*omegsave*(refindex1save*thickn1-refindex2save*thickn2-nairsave*d)/c)
    den6ref=-r2a*r2a*np.exp(j*omegsave*-1*(refindex1save*thickn1+refindex2save*thickn2+nairsave*d)/c)
    den7ref=-ra1*ra1*np.exp(j*omegsave*(refindex1save*thickn1+refindex2save*thickn2-nairsave*d)/c)
    den8ref=-r2a*r2a*np.exp(j*omegsave*(refindex1save*thickn1+nairsave*d-refindex2save*thickn2)/c)
    
    
    Zref2=ttref/(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref)
    
#    Zref1=np.where(np.isnan(Zref)==False,Zref,1e-100+1e-100*1j)
#    Zref2=np.where(np.isinf(Zref1)==False,Zref1,1e+100+1e+100*1j)
#    Zsample1=np.where(np.isnan(Zsample)==False,Zsample,1e-100+1e-100*1j)
#    Zsample2=np.where(np.isinf(Zsample1)==False,Zsample1,1e+100+1e+100*1j)		

    transfcttot=Zsample2/Zref2
    
    anglettunwrap = np.unwrap(np.angle(tt))
    anglett=anglettunwrap[-1]
    angletf = np.unwrap(np.angle((den1+den2+den3+den4+den5+den6+den7+den8))) #-2*10*np.pi
    angletfunwrap = angletf[-1]
    
    anglettrefunwrap = np.unwrap(np.angle(ttref))
    anglettref = anglettrefunwrap[-1]
    angleref = np.unwrap(np.angle(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref))
    anglerefunwrap = angleref[-1]
    #anglerefunwrap = omeg*nair*(thickn1+d+thickn2)/c
    measunwrap=np.unwrap(np.angle(meassave))
    anglemeas=measunwrap[-1]
    errorAngle2 = ((anglett - angletfunwrap + anglerefunwrap - anglettref) - anglemeas) ** 2
    errorAbs = (np.log(abs(transfcttot[-1])) - np.log(abs(meas))) ** 2
    erreur2 = (errorAngle2 + errorAbs)
    return erreur2

def errorFct3(n, k, d, frequency, meas, meassave, FP, eps_cell, thickn1, thickn2, fsave):
    """ Function to create a model giving, the frequencies, its "n", its "k", the thickness of the sample in m
    and if the Fabry-Perot effect is taken into account"""
    omeg=2*np.pi*frequency
    refindex1 = np.sqrt(eps_cell)
    refindex2 = np.sqrt(eps_cell)
    omegsave = 2*np.pi*fsave
    nair=1.0
    nairsave = np.ones(len(fsave))
    refInd = n - 1j * k

    refIndsave = np.ones(len(fsave))*refInd
    refindex1save = np.ones(len(fsave))*refindex1
    refindex2save = np.ones(len(fsave))*refindex2
    
    # Fresnel coefficients (transmission)
    t2a=2*refindex2save/(refindex2save+nairsave)
    ta1=2*nairsave/(refindex1save+nairsave)
    t1s=2*refindex1save/(refindex1save+refIndsave)
    ts2=2*refIndsave/(refindex2save+refIndsave)
    tt = t2a * ta1 * t1s * ts2
    
#    rs1=(refIndsave-refindex1save)/(refIndsave+refindex1save)
#    rs2=(refIndsave-refindex2save)/(refIndsave+refindex2save)
#    r1s=(refindex1save-refIndsave)/(refIndsave+refindex1save)
#    r2s=(refindex2save-refIndsave)/(refIndsave+refindex2save)
#    r1a=(refindex1save-nairsave)/(refindex1save+nairsave)
#    r2a=(refindex2save-nairsave)/(refindex2save+nairsave)
    rs1=(refIndsave-refindex1save)/(refIndsave+refindex1save)
    rs2=(refIndsave-refindex2save)/(refIndsave+refindex2save)
    r1s=(refindex1save-refIndsave)/(refIndsave+refindex1save)
    r2s=(refindex2save-refIndsave)/(refIndsave+refindex2save)
    r1a=(refindex1save-nairsave)/(nairsave+refindex1save)
    r2a=(refindex2save-nairsave)/(nairsave+refindex2save)
    
    den1=np.exp(j*omegsave*(refindex1save*thickn1+refIndsave*d+refindex2save*thickn2)/c)
    den2=-r1a*r1s*np.exp(j*omegsave*(refIndsave*d+refindex2save*thickn2-refindex1save*thickn1)/c)
    den3=-r1a*rs2*np.exp(j*omegsave*(refindex2save*thickn2-refIndsave*d-refindex1save*thickn1)/c)
    den4=r2a*r2a*r2s*r2s*np.exp(j*omegsave*(-refindex1save*thickn1-refindex2save*thickn2+refIndsave*d)/c)
    den5=-rs1*r2a*np.exp(j*omegsave*(refindex1save*thickn1-refindex2save*thickn2-refIndsave*d)/c)
    den6=-r2a*r2a*np.exp(j*omegsave*-1*(refindex1save*thickn1+refindex2save*thickn2+refIndsave*d)/c)
    den7=-rs1*rs1*np.exp(j*omegsave*(refindex1save*thickn1+refindex2save*thickn2-refIndsave*d)/c)
    den8=-r2a*r2s*np.exp(j*omegsave*(refindex1save*thickn1+refIndsave*d-refindex2save*thickn2)/c)
		
    Zsample2=tt/(den1+den2+den3+den4+den5+den6+den7+den8)
    #Zsample2=tt/(den1+den2)
    #Zref2=np.exp(-j*omegsave*nairsave*(thickn1+d+thickn2)/c)
    
    t1a=2*refindex1save/(refindex1save+nairsave)
    ta2=2*nairsave/(refindex2save+nairsave)
    
    ttref=t2a*ta1*t1a*ta2
    
    ra1=-r1a
    ra2=-r2a
    
    den1ref=np.exp(j*omegsave*(refindex1save*thickn1+nairsave*d+refindex2save*thickn2)/c)
    den2ref=-r1a*r1a*np.exp(j*omegsave*(nairsave*d+refindex2save*thickn2-refindex1save*thickn1)/c)
    den3ref=-r1a*ra2*np.exp(j*omegsave*(refindex2save*thickn2-nairsave*d-refindex1save*thickn1)/c)
    den4ref=r2a*r2a*r2a*r2a*np.exp(j*omegsave*(-refindex1save*thickn1-refindex2save*thickn2+nairsave*d)/c)
    den5ref=-ra1*r2a*np.exp(j*omegsave*(refindex1save*thickn1-refindex2save*thickn2-nairsave*d)/c)
    den6ref=-r2a*r2a*np.exp(j*omegsave*-1*(refindex1save*thickn1+refindex2save*thickn2+nairsave*d)/c)
    den7ref=-ra1*ra1*np.exp(j*omegsave*(refindex1save*thickn1+refindex2save*thickn2-nairsave*d)/c)
    den8ref=-r2a*r2a*np.exp(j*omegsave*(refindex1save*thickn1+nairsave*d-refindex2save*thickn2)/c)
    
    
    Zref2=ttref/(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref)
    
#    Zref1=np.where(np.isnan(Zref)==False,Zref,1e-100+1e-100*1j)
#    Zref2=np.where(np.isinf(Zref1)==False,Zref1,1e+100+1e+100*1j)
#    Zsample1=np.where(np.isnan(Zsample)==False,Zsample,1e-100+1e-100*1j)
#    Zsample2=np.where(np.isinf(Zsample1)==False,Zsample1,1e+100+1e+100*1j)		

    transfcttot=Zsample2/Zref2
    anglettunwrap = np.unwrap(np.angle(tt))
    anglett=anglettunwrap[-1]
    angletf = np.unwrap(np.angle((den1+den2+den3+den4+den5+den6+den7+den8))) #-2*10*np.pi
    angletfunwrap = angletf[-1]
    
    anglettrefunwrap = np.unwrap(np.angle(ttref))
    anglettref = anglettrefunwrap[-1]
    angleref = np.unwrap(np.angle(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref))
    anglerefunwrap = angleref[-1]
    #anglerefunwrap = omeg*nair*(thickn1+d+thickn2)/c
    measunwrap=np.unwrap(np.angle(meassave))
    anglemeas=measunwrap[-1]
    errorAngle2 = ((anglett - angletfunwrap + anglerefunwrap - anglettref) - anglemeas) ** 2
    errorAbs = (np.log(abs(transfcttot[-1])) - np.log(abs(meas))) ** 2
    erreur2 = (errorAngle2 + errorAbs)
    return erreur2, errorAbs, errorAngle2, anglett, angletfunwrap, anglerefunwrap, anglemeas

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
        pouet = lambda x: errorFct(x[0], x[1], thick, f[i], measAbsTh[i], measAngleTh[i], FP)
        res = minimize(pouet, init[i], method="Nelder-Mead", tol=1e-6)
        # cf https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
        # for other methods
        nInv[i] = res.x[0]
        kInv[i] = res.x[1]

    ref_index = j * kInv + nInv

    return ref_index

# defining the 'so-called' inverse problem to get the refractive index
def inverseProblem2(freq, fctTrans, thick, FP, init, limitUp, eps_cell, thickn1, thickn2):
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
    fsave = []

    ## The measurement
    meas = fctTrans#[(freq <= limitUp)]

    ## Inverse problem
    nInv = zeros(freq.shape[0])
    kInv = zeros(freq.shape[0])
    erreur2=zeros(freq.shape[0])
    errorAbs=zeros(freq.shape[0])
    errorAngle2=zeros(freq.shape[0])
    anglett=zeros(freq.shape[0])
    angletfunwrap=zeros(freq.shape[0])
    anglerefunwrap=zeros(freq.shape[0])
    anglemeas=zeros(freq.shape[0])
    #measAngleTh = unwrap(angle(meas))
    #measAbsTh = abs(meas)
    
    for i in range(0, f.shape[0] - 1):
        fsave=np.append(fsave,f[i])
        meassave=meas[(freq <= freq[i])]
        pouet = lambda x: errorFct2(x[0], x[1], thick, f[i], meas[i], meassave, FP, eps_cell[i], thickn1, thickn2, fsave)
        res = minimize(pouet, init[i], method="Nelder-Mead", tol=1e-6)
        
        pouet2 = errorFct3(res.x[0], res.x[1], thick, f[i], meas[i], meassave, FP, eps_cell[i], thickn1, thickn2, fsave)
        erreur2[i]=pouet2[0]
        errorAbs[i]=pouet2[1]
        errorAngle2[i]=pouet2[2]
        anglett[i]=pouet2[3]
        angletfunwrap[i]=pouet2[4]
        anglerefunwrap[i]=pouet2[5]
        anglemeas[i]=pouet2[6]
        nInv[i] = res.x[0]
        kInv[i] = res.x[1] 
        
    ref_index = 1j * kInv + nInv

    return ref_index,erreur2,errorAbs,errorAngle2,anglett,angletfunwrap,anglerefunwrap,anglemeas

def dielcal(mytransferfunction, z, myglobalparameters, FP):
    global zz, mytransferfunction2


    """ 
    Takes variables from the the input and uses the inverseProblem
    The first part is the initialization and the last line is the optimisation
    """
    mytransferfunction2 = np.copy(mytransferfunction)
    zz = np.copy(z)
    ref_index = np.ones(len(myglobalparameters.w))
    myangleTF = np.unwrap(np.angle(mytransferfunction))

    for compt in range(0, 3):  # the max of this loop was choosen to optimize the effective index for a 250micron thick quartz substrate it gives good result at 1/1000 but shows weird convergeance (larger is not better so this may had to be changed with the sample)
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w * zz * (ref_index) / c)
        myangleTrans = np.nan_to_num(np.angle(tt))  # ici il y a un truc a bien regarder
        phase = np.unwrap(myangleTF + myangleTrans)  # we are taking the phase from the transfer function
        naprox = 1 + abs(phase * c / zz / myglobalparameters.w)  # this gives the real part of the index
        ref_index = naprox + j * (ref_index.imag)  # this gives the real part of the index
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w *zz* (ref_index) / c)
        kaprox = -np.nan_to_num(c / zz / myglobalparameters.w * np.log(abs(mytransferfunction) / (tt)))
        ref_index = j * kaprox.real + ref_index.real
    ref_index[0] = 1

    ref_index2 = inverseProblem( myglobalparameters.w / 2 / np.pi, mytransferfunction2, zz, FP, ref_index, 6e12)

    return np.array(ref_index2 ** 2)

def dielcal2(mytransferfunction, z, myglobalparameters, eps_cell, thickn1, thickn2,FP):
    global zz, mytransferfunction2


    """ 
    Takes variables from the the input and uses the inverseProblem
    The first part is the initialization and the last line is the optimisation
    """
    mytransferfunction2 = np.copy(mytransferfunction)
    zz = np.copy(z)
    ref_index = np.ones(len(myglobalparameters.w))+j*np.zeros(len(myglobalparameters.w))
    myangleTF = np.unwrap(np.angle(mytransferfunction))
    
    refindex1=np.sqrt(eps_cell)
    refindex2=np.sqrt(eps_cell)
    omeg=myglobalparameters.w

    for compt in range(0, 1):  # the max of this loop was choosen to optimize the effective index for a 250micron thick quartz substrate it gives good result at 1/1000 but shows weird convergeance (larger is not better so this may had to be changed with the sample)
        t2a=2*refindex2/(refindex2+1)
        ta1=2/(refindex1+1)
        t1s=2*refindex1/(refindex1+ref_index)
        ts2=2*ref_index/(refindex2+ref_index)
        tt = t2a * ta1 * t1s * ts2
        
        rs1=(ref_index-refindex1)/(ref_index+refindex1)
        rs2=(ref_index-refindex2)/(ref_index+refindex2)
        r1s=(refindex1-ref_index)/(ref_index+refindex1)
        r2s=(refindex2-ref_index)/(ref_index+refindex2)
        r1a=(refindex1-1)/(refindex1+1)
        r2a=(refindex2-1)/(refindex2+1)
        
        den1=np.exp(j*omeg*(refindex1*thickn1+ref_index*zz+refindex2*thickn2)/c)
        den2=-r1a*r1s*np.exp(j*omeg*(ref_index*zz+refindex2*thickn2-refindex1*thickn1)/c)
        den3=-r1a*rs2*np.exp(j*omeg*(refindex2*thickn2-ref_index*zz-refindex1*thickn1)/c)
        den4=r2a*r2a*r2s*r2s*np.exp(j*omeg*(-refindex1*thickn1-refindex2*thickn2+ref_index*zz)/c)
        den5=-rs1*r2a*np.exp(j*omeg*(refindex1*thickn1-refindex2*thickn2-ref_index*zz)/c)
        den6=-r2a*r2a*np.exp(j*omeg*-1*(refindex1*thickn1+refindex2*thickn2+ref_index*zz)/c)
        den7=-rs1*rs1*np.exp(j*omeg*(refindex1*thickn1+refindex2*thickn2-ref_index*zz)/c)
        den8=-r2a*r2s*np.exp(j*omeg*(refindex1*thickn1+ref_index*zz-refindex2*thickn2)/c)
        		
        Zsample2=tt/(den1+den2+den3+den4+den5+den6+den7+den8)
        
        t1a=2*refindex1/(refindex1+1)
        ta2=2*1/(refindex2+1)
        
        ttref=t2a*ta1*t1a*ta2
        
        ra1=-r1a
        ra2=-r2a
        
        den1ref=np.exp(j*omeg*(refindex1*thickn1+1*zz+refindex2*thickn2)/c)
        den2ref=-r1a*r1a*np.exp(j*omeg*(1*zz+refindex2*thickn2-refindex1*thickn1)/c)
        den3ref=-r1a*ra2*np.exp(j*omeg*(refindex2*thickn2-1*zz-refindex1*thickn1)/c)
        den4ref=r2a*r2a*r2a*r2a*np.exp(j*omeg*(-refindex1*thickn1-refindex2*thickn2+1*zz)/c)
        den5ref=-ra1*r2a*np.exp(j*omeg*(refindex1*thickn1-refindex2*thickn2-1*zz)/c)
        den6ref=-r2a*r2a*np.exp(j*omeg*-1*(refindex1*thickn1+refindex2*thickn2+1*zz)/c)
        den7ref=-ra1*ra1*np.exp(j*omeg*(refindex1*thickn1+refindex2*thickn2-1*zz)/c)
        den8ref=-r2a*r2a*np.exp(j*omeg*(refindex1*thickn1+1*zz-refindex2*thickn2)/c)
        
        
        #Zref2=1/np.exp(j*omeg*(1*thickn1+1*zz+1*thickn2)/c)
        Zref2=ttref/(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref)
        transfctth=Zsample2/Zref2


        myangleTrans = np.nan_to_num(np.unwrap(np.angle(transfctth)))  # ici il y a un truc a bien regarder
        myangleTF = np.unwrap(np.nan_to_num(np.angle(mytransferfunction)))#-np.nan_to_num(np.unwrap(np.angle(Spulseinit)))
        phase = (-myangleTF + myangleTrans)  # we are taking the phase from the transfer function
        naprox = 1+abs(phase * c / (zz * omeg))  # this gives the real part of the index
        #naprox = 2*np.ones(len(myglobalparameters.w))
        ref_index = naprox + j * (ref_index.imag)  # this gives the real part of the index
        
        t2a=2*refindex2/(refindex2+1)
        ta1=2/(refindex1+1)
        t1s=2*refindex1/(refindex1+ref_index)
        ts2=2*ref_index/(refindex2+ref_index)
        tt = t2a * ta1 * t1s * ts2
        
        rs1=(ref_index-refindex1)/(ref_index+refindex1)
        rs2=(ref_index-refindex2)/(ref_index+refindex2)
        r1s=(refindex1-ref_index)/(ref_index+refindex1)
        r2s=(refindex2-ref_index)/(ref_index+refindex2)
        r1a=(refindex1-1)/(refindex1+1)
        r2a=(refindex2-1)/(refindex2+1)
        
        den1=np.exp(j*omeg*(refindex1*thickn1+ref_index*zz+refindex2*thickn2)/c)
        den2=-r1a*r1s*np.exp(j*omeg*(ref_index*zz+refindex2*thickn2-refindex1*thickn1)/c)
        den3=-r1a*rs2*np.exp(j*omeg*(refindex2*thickn2-ref_index*zz-refindex1*thickn1)/c)
        den4=r2a*r2a*r2s*r2s*np.exp(j*omeg*(-refindex1*thickn1-refindex2*thickn2+ref_index*zz)/c)
        den5=-rs1*r2a*np.exp(j*omeg*(refindex1*thickn1-refindex2*thickn2-ref_index*zz)/c)
        den6=-r2a*r2a*np.exp(j*omeg*-1*(refindex1*thickn1+refindex2*thickn2+ref_index*zz)/c)
        den7=-rs1*rs1*np.exp(j*omeg*(refindex1*thickn1+refindex2*thickn2-ref_index*zz)/c)
        den8=-r2a*r2s*np.exp(j*omeg*(refindex1*thickn1+ref_index*zz-refindex2*thickn2)/c)
        		
        Zsample2=tt/(den1+den2+den3+den4+den5+den6+den7+den8)
        
        t1a=2*refindex1/(refindex1+1)
        ta2=2*1/(refindex2+1)
        
        ttref=t2a*ta1*t1a*ta2
        
        ra1=-r1a
        ra2=-r2a
        
        den1ref=np.exp(j*omeg*(refindex1*thickn1+1*zz+refindex2*thickn2)/c)
        den2ref=-r1a*r1a*np.exp(j*omeg*(1*zz+refindex2*thickn2-refindex1*thickn1)/c)
        den3ref=-r1a*ra2*np.exp(j*omeg*(refindex2*thickn2-1*zz-refindex1*thickn1)/c)
        den4ref=r2a*r2a*r2a*r2a*np.exp(j*omeg*(-refindex1*thickn1-refindex2*thickn2+1*zz)/c)
        den5ref=-ra1*r2a*np.exp(j*omeg*(refindex1*thickn1-refindex2*thickn2-1*zz)/c)
        den6ref=-r2a*r2a*np.exp(j*omeg*-1*(refindex1*thickn1+refindex2*thickn2+1*zz)/c)
        den7ref=-ra1*ra1*np.exp(j*omeg*(refindex1*thickn1+refindex2*thickn2-1*zz)/c)
        den8ref=-r2a*r2a*np.exp(j*omeg*(refindex1*thickn1+1*zz-refindex2*thickn2)/c)
        
        
        #Zref2=1/np.exp(j*omeg*(1*thickn1+1*zz+1*thickn2)/c)
        Zref2=ttref/(den1ref+den2ref+den3ref+den4ref+den5ref+den6ref+den7ref+den8ref)
        
        transfctth=Zsample2/Zref2
        
        kaprox = -np.nan_to_num(c*(1/zz)*2*np.log(abs(mytransferfunction/transfctth))/(2*omeg))
        #kaprox=np.zeros(len(omeg))
        ref_index = j * kaprox.real + ref_index.real
    #ref_index[0] = 1

    #print(ref_index)
#    eps =2.5*np.ones(len(myglobalparameters.w))
#    delta_eps=35
#    tau=3e-12
#    eps=eps+delta_eps/(1E0+j*tau*myglobalparameters.w)
#    ref_index = np.sqrt(eps)
    
#    plt.figure('ref_index',figsize=(12,8))
#    plt.plot(myglobalparameters.freq,np.real(ref_index),'b-')
#    plt.plot(myglobalparameters.freq,np.imag(ref_index),'r-')
    
    test = inverseProblem2(myglobalparameters.w / 2 / np.pi, mytransferfunction2, zz, 1, ref_index, 6e12, eps_cell, thickn1, thickn2)
    ref_index2 = test[0]
    erreur2=test[1]
    errorAbs=test[2]
    errorAngle2=test[3]
    anglett=test[4]
    angletfunwrap=test[5]
    anglerefunwrap=test[6]
    anglemeas=test[7]   
    return np.array(ref_index2 ** 2), erreur2,errorAbs,errorAngle2,anglett,angletfunwrap,anglerefunwrap,anglemeas
