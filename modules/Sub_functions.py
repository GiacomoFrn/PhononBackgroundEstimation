import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy import interpolate

from modules.fitQ import fitQdistrib, fit_func

def readMap( data_df ):
    #saving df columns in array
    Q_arr = data_df["Q"].to_numpy() 
    E_arr = data_df["E"].to_numpy() 
    S_arr = data_df["S"].to_numpy() 
    #S_E_arr = data_df["S_E"].to_numpy()

    #delete duplicate values due to grid structure
    Q_arr = np.unique(Q_arr)
    E_arr = np.unique(E_arr)
    
    return Q_arr, E_arr, S_arr

def Qcut( Q_arr, E_arr, S_arr, Q, dQ, Emin, Emax ):
    S_arr2D = S_arr.reshape(len(Q_arr), len(E_arr))
    S_arr2D = S_arr2D.transpose()
    S_arr2D = np.flipud(S_arr2D)
    
    S_int = []
    for E_idx in range(S_arr2D.shape[0]):
        S_sum = 0
        for Q_idx in range(S_arr2D.shape[1]):
            if Q_arr[Q_idx]>(Q-dQ) and Q_arr[Q_idx]<(Q+dQ) :
                S_sum += S_arr2D[E_idx][Q_idx]
        S_int.append(S_sum)  
    S_int = np.flip(S_int)
    
    minimum = abs(Q_arr-Q).min()
    idx_Q = np.where(abs(Q_arr-Q) == minimum)[0][0]

    S_Q = []
    for E_idx in range(S_arr2D.shape[0]):
        S_Q.append(S_arr2D[E_idx][idx_Q])
    S_Q = np.flip(S_Q)
    
    ratio = []
    for i in range(len(S_Q)):
        if E_arr[i] < Emax and E_arr[i]>Emin :   #doesn't make too much sense for me but it works
            if not (pd.isna(S_int[i]) or pd.isna(S_Q[i])):
                ratio.append(S_int[i]/S_Q[i])
    ratio = np.array(ratio)
    
    return S_int, ratio

def S_splined(E, E_arr, S_arr):
    tck = interpolate.splrep(E_arr, S_arr)
    return float(interpolate.splev(E, tck))

def intensity(Q, S, Q_arr, E_arr, scaling_factors):
    idx_Q = np.where(Q_arr == Q)[0][0]
    intensity_ =  S*scaling_factors[idx_Q]
    return intensity_

def PhononMask( Q_arr, E_arr, S_arr, S_int,  Q, Emin, Emax, popt ):
    scaling_factors = fit_func(Q_arr, *popt)
    minimum = abs(Q_arr-Q).min()
    idx = np.where(abs(Q_arr-Q) == minimum)[0][0]

    scaling_factors /= scaling_factors[idx]
    
    S_mask = []

    E_lim = []
    S_int_lim = []
    for E, S in zip(E_arr, S_int):
        if  (E>Emin and E<Emax):
            if not (pd.isna(E) or pd.isna(S)):
                E_lim.append(E)
                S_int_lim.append(S)
        
    E_lim = np.array(E_lim)
    S_int_lim = np.array(S_int_lim)

        
    for Q in Q_arr:
        for E in E_arr:
            if  (E>Emin and E<Emax):
                S = S_splined(E, E_lim, S_int_lim)
                S_mask.append(intensity(Q, S, Q_arr, E_lim, scaling_factors))
            else :
                S_mask.append(0)
                
    return S_mask