# =============================================================================
# Aditya_code
# =============================================================================
import sympy as sp
import numdifftools as nd
import pandas as pd
import pyarrow as pa
# from tqdm import *
import pyarrow.csv  as csv_
from pathlib import Path as path_

ROOT_DIR = path_(__file__).parent
# filename = path_(ROOT_DIR).joinpath('errors.csv')
filename = path_(ROOT_DIR)

# ============================================================================ #
#                        sympy error function definition                       #
# ============================================================================ #
n, k, w, d, C1, C2 = sp.symbols('n k w d C1 C2', real=True)

# # Define complex-valued function
c = 2.998e8
A1 = (4*n**3+4*n*k**2+8*n**2+8*k**2+4*n)/((k**2+(1+n)**2)**2)
B1 = (4*k*(-n**2-k**2+1))/((k**2+(1+n)**2)**2)
D1 = w*d*k/c
D2 = w*d*(n-1)/c
f =  C1**2 + C2**2 - 2*(C1*A1 - C2*B1)*sp.exp(-D1)*sp.cos(D2) + 2*(C1*B1 + A1*C2)*sp.exp(-D1)*sp.sin (D2) + A1**2*sp.exp(-2*D1) + B1**2*sp.exp(-2*D1)
Hessian = sp.hessian(f,(n,k))
# ---------------------------------------------------------------------------- #


# =============================================================================
# Aditya_code
# =============================================================================

###############################################################################
###############################################################################
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
import csts as Csts

###############################################################################
###############################################################################
j = 1j
c = 2.998e8

name = Csts.FileName
###############################################################################

# defining the Error function
def errorFct(n, k, d, frequency, measAmp, measAngle, FP, eqs = "log",print_bool = False):
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
        # FP_effect = 1 / (1 - ((refInd - nair) / (refInd + nair)) ** 2 * exp(-2 * 1j * refInd * omeg * d / c))
        FP_effect = 1 / (1 - np.square((refInd - nair) / (refInd + nair)) * exp(-2 * 1j * refInd * omeg * d / c))
        z = Taf * Tfa * Tpe * FP_effect
    else:
        z = Taf * Tfa * Tpe 

    angleTaf = angle(Taf)
    angleTfa = angle(Tfa)
    angleTfe = -(n - nair) * omeg * d / c
    angleFP = angle(FP_effect)

    # print(f"z :  {z}")
    # print(f"measAmp :  {measAmp}")
    
    #à utiliser en premier lieu
    if eqs == "exp":
        ThZ = abs(z)*abs(exp(j*(angleTaf+angleTfa+angleTfe+angleFP)))
        measZ = measAmp*abs(exp(j*measAngle))
        erreur = abs((ThZ-measZ))**2
    
    #à utiliser si la version en exp ne marche pas (partie réelle de l'indice de réfraction inversée)
    elif eqs == "log":
        ThZ = (np.log(abs(z))+1j*(angleTaf+angleTfa+angleTfe+angleFP))
        measZ = ((log(measAmp))+1j*measAngle)
        erreur = abs((ThZ-measZ))**2
    
    # print_bool = True
    if print_bool:
        print(f"n : {n}")
        print(f"k : {k}")
        print(f"err = {erreur}")

    #version antérieure, pas bon car séparait la formule en deux
    #errorAngle = (angleTaf + angleTfa +temp_ angleTfe + angleFP - measAngle) ** 2
    #errorAbs = (log(abs(z)) - log(measAmp)) ** 2
    #erreur = abs(errorAngle + errorAbs)

    return erreur

# ############## try to filter wrong n and k values for hessian ############## #
def errorFct_hess(n, k, n_ini, k_ini, d, frequency, measAmp, measAngle, FP, print_bool = False):
    """ Function to create a model giving, the frequencies, its "n", its "k", the thickness of the sample in m
    and if the Fabry-Perot effect is taken into account"""
    
    # if n < n_ini or n < 0:
    # if  n < 0 :
        # # n = n_ini
        # n = abs(n)
    # if k < 0 :
        # # k = k_ini
        # k = abs(k)
    
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
        # FP_effect = 1 / (1 - ((refInd - nair) / (refInd + nair)) ** 2 * exp(-2 * 1j * refInd * omeg * d / c))
        FP_effect = 1 / (1 - np.square((refInd - nair) / (refInd + nair)) * exp(-2 * 1j * refInd * omeg * d / c))
        z = Taf * Tfa * Tpe * FP_effect
    else:
        z = Taf * Tfa * Tpe 

    angleTaf = angle(Taf)
    angleTfa = angle(Tfa)
    angleTfe = -(n - nair) * omeg * d / c
    angleFP = angle(FP_effect)

    #à utiliser en premier lieu
    # ThZ = abs(z)*abs(exp(j*(angleTaf+angleTfa+angleTfe+angleFP)))
    # measZ = measAmp*abs(exp(j*measAngle))
    # erreur = abs((ThZ-measZ))**2
    
    # erreur = np.square(np.abs(z-measAmp))
    
    #à utiliser si la version en exp ne marche pas (partie réelle de l'indice de réfraction inversée)
    ThZ = (np.log(abs(z))+1j*(angleTaf+angleTfa+angleTfe+angleFP))
    measZ = ((log(measAmp))+1j*measAngle)
    erreur = abs((ThZ-measZ))**2
    
    if print_bool:
        print(f"n : {n}")
        print(f"k : {k}")
        print(f"err = {erreur}")
        
        # print(f"refIndex : {refInd}")
        # print(f"THz={ThZ}")
        # print(f"measz = {measZ}")

    #version antérieure, pas bon car séparait la formule en deux
    #errorAngle = (angleTaf + angleTfa + angleTfe + angleFP - measAngle) ** 2
    #errorAbs = (log(abs(z)) - log(measAmp)) ** 2
    #erreur = abs(errorAngle + errorAbs)

    return erreur
    

def errorFct_TF(n, k, d, frequency, ftrans_meas, FP, print_bool = False):
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
        # FP_effect = 1 / (1 - ((refInd - nair) / (refInd + nair)) ** 2 * exp(-2 * 1j * refInd * omeg * d / c))
        FP_effect = 1 / (1 - np.square((refInd - nair) / (refInd + nair)) * np.exp(-2 * 1j * refInd * omeg * d / c))
        z = Taf * Tfa * Tpe * FP_effect
    else:
        z = Taf * Tfa * Tpe 

    
    erreur = np.square(np.abs(z-ftrans_meas))
    
    if print_bool:
        print(f"n : {n}")
        print(f"k : {k}")
        print(f"err_trans = {erreur}")
    

    return erreur

##Aditya_code

# def errorFct2(n, k, d, f, c1, c2):
    # 
    # w = 2*np.pi*f
    # a1 = (4*n**3+4*n*k**2+8*n**2+8*k**2+4*n)/((k**2+(1+n)**2)**2)
    # b1 = (4*k*(-n**2-k**2+1))/((k**2+(1+n)**2)**2)
    # d1 = w*d*k/c
    # d2 = w*d*(n-1)/c
    # erreur2 =  c1**2 + c2**2 - 2*(c1*a1 - c2*b1)*np.exp(-d1)*np.cos(d2) + 2*(c1*b1 + a1*c2)*np.exp(-d1)*np.sin(d2) + a1**2*np.exp(-2*d1) + b1**2*np.exp(-2*d1)
    # 
    # print(f"n_r : {n}")
    # print(f"k_r : {k}")
    # print(f"erreur_r : {erreur2}")

    # return erreur2


##Aditya_code

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
    # nnn = np.ones(init.shape)
    # init = np.transpose([nnn, init.imag])
    
    f = freq[(freq <= limitUp)]

    ## The measurement
    meas = fctTrans[(freq <= limitUp)]

    ## Inverse problem
    nInv = zeros(freq.shape[0])
    kInv = zeros(freq.shape[0])
    
    # =============================== Hessian matix ============================== #
    Hessian_nd_c = np.empty((freq.shape[0]),dtype=matrix)
    Gradient_nd_c = np.empty((freq.shape[0]),dtype=matrix)
    # Hessian_nd_r = np.empty((freq.shape[0]),dtype=matrix)
    # Hessian_sp_r = np.empty((freq.shape[0]),dtype=matrix)
    
    err_n  = np.zeros((freq.shape[0]))
    err_k  = np.zeros((freq.shape[0]))
    
    diag_n = np.zeros((freq.shape[0]))
    diag_k = np.zeros((freq.shape[0]))
    
    inv_n = np.zeros((freq.shape[0]))
    inv_k = np.zeros((freq.shape[0]))
    
    hess_n = np.zeros((freq.shape[0]))
    hess_k = np.zeros((freq.shape[0]))
    
    err_n_bfgs = np.zeros((freq.shape[0]))
    err_k_bfgs = np.zeros((freq.shape[0]))
    
    # err2_n  = np.empty((freq.shape[0]))
    # err2_k  = np.empty((freq.shape[0]))
    # err3_n  = np.empty((freq.shape[0]))
    # err3_k  = np.empty((freq.shape[0]))
    # ---------------------------------------------------------------------------- #
    
    measAngleTh = unwrap(angle(meas)) 
    # measAngleTh = unwrap(angle(meas)) + 2 * np.pi
    measAbsTh = abs(meas)
    
    for i in range(0, f.shape[0] - 1):
        # pouet = lambda x: errorFct(x[0], x[1], thick, f[i], measAbsTh[i], measAngleTh[i], FP)
        # res = minimize(pouet, init[i], method="Nelder-Mead", tol=1e-6)
        # res = minimize(pouet, init[i], method="SLSQP")
        
        # # res = minimize(err_function_log, init[i], method="BFGS", tol=1e-6)
        err_function_min = lambda x: errorFct_TF(x[0], x[1], thick, f[i], fctTrans[i], FP)
        res = minimize(err_function_min, init[i], method="Nelder-Mead", tol=1e-6)
        
        err_function_log = lambda x: errorFct(x[0], x[1], thick, f[i], measAbsTh[i], measAngleTh[i], FP, "log")
        
        # res = minimize(err_function_log, init[i], method="Nelder-Mead", tol=1e-6)
        err_n_bfgs[i] = 0
        err_k_bfgs[i] = 0
                
        # res = minimize(err_function_log, init[i], method="BFGS", tol=1e-6)
        # if i==0:
            # res = minimize(err_function_log, init[i], method="BFGS", tol=1e-6)
        # else:
            # res = minimize(err_function_log, [nInv[i-1],kInv[i-1]], method="BFGS", tol=1e-6)

        
        # err_n_bfgs[i] = np.sqrt(res["hess_inv"][0][0])
        # err_k_bfgs[i] = np.sqrt(res["hess_inv"][1][1])
        
        
        
        # # err_funtion_log_minimzied = errorFct(res['x'][0], res['x'][1], thick, f[i], measAbsTh[i], measAngleTh[i], FP,print_bool=True)
        # err_fctrans = errorFct_TF(res['x'][0], res['x'][1], thick, f[i], fctTrans[i], FP,)
        
        err_function_trans = lambda x: errorFct_TF(x[0], x[1], thick, f[i], fctTrans[i], FP)
        # res = minimize(err_function_trans, init[i], method="BFGS", tol=1e-6)
        res = minimize(err_function_trans, init[i], method="BFGS", tol=1e-6)
        
        print_bool_res = False
        
        if print_bool_res:
            print(f"###################") 
            print(f"freq : {f[i]/1e12}")
            print(f"init : {init[i]}")
            print(f"res : {res}")
            print(f"minimizing...")
            print(f"Status : {res['message']}")
            # print(f" err_n_bfgs : {err_n_bfgs[i]}")
            # print(f" err_k_bfgs : {err_k_bfgs[i]}")
            print(f"Total Evaluations: {res['nfev']}")
            print(f"n : {res['x'][0]}")
            print(f"k : {res['x'][1]}")
            # print(f"erreur : {err_funtion_log_minimzied}")
            # print(f"erreur : {err_fctrans}")
            print(f"====================") 
        
        # cf https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
        # for other methods
        nInv[i] = res.x[0]
        kInv[i] = res.x[1]
    
    #TOVERIFY
    for i in range(0, f.shape[0] - 1):
        # puick = errorFct(res.x[0],res.x[1],thick,f[i],measAbsTh[i], measAngleTh[i], FP)
        # puick = errorFct(nInv[i],kInv[i],thick,f[i],measAbsTh[i], measAngleTh[i], FP)
        
        ##Aditya_code

        # ========================== numdifftools in C space ========================= #
        # G_numdiff_C = nd.Gradient(pouet)([res.x[0], res.x[1]])
        
        # H_Numdiff_C = nd.Hessian(pouet, method = "central", order =2)([res.x[0], res.x[1]])
        # H_Numdiff_C = nd.Hessian(pouet)([res.x[0], res.x[1]])
        
        
        # ############################################################################ #
        if print_bool_res:
            print(f"###################")
            print(f"replacing")
            print(f"freq: {f[i]/1e12}")
            print(f"n : {nInv[i]}")
            print(f"k : {kInv[i]}")
        
        try:
            step_val = 1e-6 * (1e-3 / thick)
            # print(f"step : {step_val}")
            # # pouetpouet = lambda x: errorFct(x[0], x[1], thick, f[i], measAbsTh[i], measAngleTh[i], FP,print_bool=True)
            # pouetpouet = lambda x: errorFct_hess(x[0], x[1], nInv[i], kInv[i],  thick, f[i], measAbsTh[i], measAngleTh[i],FP,print_bool=True)
            # H_Numdiff_C = nd.Hessian(pouetpouet, step=1e-6)([nInv[i] , kInv[i]])
            err_fct_hess = lambda x: errorFct_TF(x[0], x[1],  thick, f[i], fctTrans[i],FP,print_bool=True)
            H_Numdiff_C = nd.Hessian(err_fct_hess, step=step_val)([nInv[i] , kInv[i]])
            Hessian_nd_c[i] = H_Numdiff_C
            inv_hess = np.linalg.inv(H_Numdiff_C)
            diag_inv_hess = np.diag(inv_hess)
            err = np.sqrt(diag_inv_hess)
            # print(f"hess_c = {H_Numdiff_C}")
            # print(f"inv_hess_c = {inv_hess}")
            if print_bool_res:
                print(f"diag_inv_hess_c = {diag_inv_hess}")
                print(f"====================")
        except Exception as e:
            print(e)
            inv_hess = np.zeros((2,2))
            diag_inv_hess = np.zeros(2)
            err = np.zeros(2)
            print(f"diag_inv_hess_c = {diag_inv_hess}")
            print(f"====================")
        # ############################################################################ #
        
        
        # det = np.linalg.det(H_Numdiff_C)
        # print(H_Numdiff_C)
        
        # Gradient_nd_c[i] = G_numdiff_C
        
        
        # err = np.sqrt(np.diag(np.linalg.inv(H_Numdiff_C)))
        
        # ############################################################################ #
        diag_n[i] = diag_inv_hess[0]
        diag_k[i] = diag_inv_hess[1]
    
        err_n[i] = err[0]
        err_k[i] = err[1]
        
        hess_n[i] = H_Numdiff_C[0][0] 
        hess_k[i] = H_Numdiff_C[1][1] 
        
        inv_n[i] = inv_hess[0][0]
        inv_k[i] = inv_hess[1][1]
        # ############################################################################ #
        
        # inv_hess_nd_c = np.linalg.inv(H_Numdiff_C)
        
        # print(f"inv nd hess = {np.linalg.inv(H_Numdiff_C)}")
        # print(f"nd hess C =  {H_Numdiff_C}")
        # print(f"inv nd hess = {inv_hess_nd_c}")
        # ---------------------------------------------------------------------------- #
    
        # ========================== numdifftools in R space ========================= #
        # pouet2 = lambda x: errorFct2(x[0],x[1],thick,f[i],np.real(measAbsTh[i]),np.imag(measAbsTh[i]))

        # H_Numdiff_R = nd.Hessian(pouet2)([nInv[i], kInv[i]])
        
        # inv_hess_r = np.linalg.inv(H_Numdiff_R)
        # diag_inv_hess_r = np.diag(inv_hess_r)
        
        
        
        # print(f"hess_r = {diag_inv_hess_r}")
        # ############################################################################ #
        
        
        # inv_hess_nd_r = np.linalg.inv(H_Numdiff_R)

        # err2 = np.sqrt(np.diag(np.linalg.inv(H_Numdiff_R)))
        # err2_n[i] = err2[0]
        # err2_k[i] = err2[1]
        # # print(f"nd hess R =  {H_Numdiff_R}")
        # # print(f"inv nd hess R= {inv_hess_nd_r}")
        # # print(f"error nd Hess C = {err2}")

        # Hessian_nd_r[i] = H_Numdiff_R
        # ---------------------------------------------------------------------------- #
    
    
    
        # ============================= sympy in R space ============================= #
        # values = {n: nInv[i], k: kInv[i], w: 2*np.pi*f[i], d: thick, C1: np.real(measAbsTh[i]), C2: np.imag(measAbsTh[i])}
        # H_sympy_R = Hessian.subs(values)
        
        # H_sympy_R = np.array(H_sympy_R).astype(np.float64)
        # Hessian_sp_r[i]= H_sympy_R
        # inv_hess_sp_r = np.linalg.inv(H_sympy_R)
        # err3 = np.sqrt(np.diag(np.linalg.inv(H_sympy_R)))
        # err3_n[i] = err3[0]
        # err3_k[i] = err3[1]
        # # print(f"sp hess R =  {H_sympy_R}")
        # # print(f"inv sp hess R= {inv_hess_sp_r}")
        # # print(f"error nd Hess C = {err3}")
        # ---------------------------------------------------------------------------- #
        
        ##Aditya_code

    ref_index = j * kInv + nInv

    err_n = np.nan_to_num(err_n)
    err_k = np.nan_to_num(err_k)

    # indexes = pd.DataFrame([nInv, kInv, err_n, err_k], columns=["n", "k", "err_n", "err_k"])
    # indexes = pd.DataFrame({"n": nInv, "k": kInv, "err_n" : err_n, "err_k" : err_k})
    # print(indexes)
    # table = pa.Table.from_pandas(indexes)
    
    # ############################################################################ #
    # table = pa.Table.from_arrays([freq/1e12,nInv,err_n, kInv,err_k],names=["freq","n","err_n", "k", "err_k"])
    table = pa.Table.from_arrays([freq/1e12,nInv,hess_n,inv_n,diag_n,err_n,err_n_bfgs, kInv,hess_k,inv_k,diag_k,err_k, err_k_bfgs],names=["freq","n","hess_n","inv_n","diag_n","err_n", "err_n_bfgs", "k","hess_k","inv_k","diag_k", "err_k", "err_k_bfgs"])
    
    err_filename = path_(Csts.FileName).name
    err_filename = filename.joinpath(f"errors_{err_filename}")
    
    options = csv_.WriteOptions(include_header=True,delimiter='\t')
    csv_.write_csv(table,err_filename,options)
    # ############################################################################ #

    # ref_index = j * kInv + nInv
    # print(f"filename = {Csts.FileName}")
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

    for compt in range(0, 3):  # the max of this loop was choosen to optimize the effective index for a 250micron thick quartz substrate it gives good result at 1/1000 but shows weird convergeance (larger is not better so this may had to be changed with the sample)
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w * (ref_index) / c * zz)
        myangleTrans = np.nan_to_num(np.angle(tt))  # ici il y a un truc a bien regarder
        phase = np.unwrap(myangleTF + myangleTrans) # we are taking the phase from the transfer function
        
        naprox = 1 + abs(phase * c / zz / myglobalparameters.w)  # this gives the real part of the index
        
        ref_index = naprox + j * (ref_index.imag)  # this gives the real part of the index
        tt = 4 * ref_index / (1 + ref_index) ** 2
        propa = np.exp(j * myglobalparameters.w * (ref_index) / c * zz)
        kaprox = -np.nan_to_num(c / z / myglobalparameters.w * np.log(abs(mytransferfunction) / (tt)))
        ref_index = j * kaprox.real + ref_index.real
    ref_index[0] = 1

    ref_index2 = inverseProblem(myglobalparameters.w / 2 / np.pi, mytransferfunction2, zz, FP, ref_index, 6e12)

    return np.array(ref_index2 ** 2)
