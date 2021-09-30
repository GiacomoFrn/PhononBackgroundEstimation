import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit

#fit function of Escan
def fit_func(x, a, b,c):
    return a+b*x**2*np.exp(-c*x**2)
#general Qdistrib fitting from Escan
def fitQdistrib( mantid_df, minQ1, maxQ1, minQ2=0, maxQ2=0, minQ3=0, maxQ3=0, minQ4=0, maxQ4=0):
    Q_arr = mantid_df["X"].to_numpy()  
    S_arr = mantid_df["Y"].to_numpy()  
    S_err = mantid_df["E"].to_numpy()
    
    #selecting only part of the data
    Q_fit = []
    S_fit = []
    S_err_fit = []
    for Q, S, Serr in zip(Q_arr, S_arr, S_err):
        if (Q> minQ1 and Q<maxQ1) and (Q<minQ2 or Q>maxQ2) and (Q<minQ3 or Q>maxQ3) and (Q<minQ4 or Q>maxQ4) :                             
            Q_fit.append(Q)
            S_fit.append(S)
            S_err_fit.append(Serr)
            
    #estimate fit parameters
    guessP = [0.053978, 0.00274,  0.01099]
    popt, pcov = curve_fit(fit_func, Q_fit, S_fit, guessP, S_err_fit, bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))
    
    #calculate estimated S values
    S_est = []
    S_ex = [] #only for continuosly plotting
    Q_ex = np.linspace(minQ1,maxQ1,1000)
    for Q in Q_fit:
        S_est.append(fit_func(Q, *popt))
    for Q in Q_ex:
        S_ex.append(fit_func(Q, *popt))
        
    #residuals
    res = []
    for Q, S in zip(Q_fit, S_fit):
        res.append(S-fit_func(Q, *popt))
        
    return popt, pcov, Q_arr, S_arr, S_err, Q_ex, S_ex, Q_fit, res, S_err_fit