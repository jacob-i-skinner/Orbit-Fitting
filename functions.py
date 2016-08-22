import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lombscargle


def alteredRV(x, K, e, w, T, P, y): #function generates RV values plot from given parameters
    check = 1
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2) #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0 = E1
        M0 = E0 - e*np.sin(E0)
        E1 = E0 +(M-M0)/(1-e*np.cos(E0))
        if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p = ((K)*(np.cos(nu+w) + (e*np.cos(w)))+y)
    return p
    
def alteredNoERV(x, K, T, P, y): #function generates RV values plot from given parameters
    check = 1
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0 = E1
        M0 = E0
        E1 = E0 +(M-M0)
        if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p = K*np.cos(nu)+y
    return p

#function generates RV values from given parameters
def RV(x, mass_ratio, parameters):
    check = 1    
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2) #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0    = E1
        M0    = E0 - e*np.sin(E0)
        E1    = E0 +(M-M0)/(1-e*np.cos(E0))
        if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p, s = (K*(np.cos(nu+w) + (e*np.cos(w)))+y), ((-K/mass_ratio)*(np.cos(nu+w) + (e*np.cos(w)))+y)
    return p, s

#a version of the RV plotter that is used if e is sufficiently close to zero
def noERV(x, mass_ratio, parameters): #function generates RV values plot from given parameters
    check = 1    
    K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0    = E1
        M0    = E0
        E1    = E0 +(M-M0)
        if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p, s = K*np.cos(nu)+y, (-K/mass_ratio)*np.cos(nu)+y
    return p, s

#This periodogram function was taken from Jake Vanderplas' article "Fast Lomb-Scargle Periodograms in Python"
def periodogram(x, rv, f, max_period):
    x = np.array(x)
    rv = np.array(rv)
# lower limit of periods range set to one hour
#    delta_x = np.inf 
#    for i in range(0, len(x)-2):
#        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
#            delta_x = x[i+1]-x[i]
    periods = np.linspace(0.04167, max_period, num = f)

    # convert period range into frequency range
    ang_freqs = 2 * np.pi / periods

    # compute the (unnormalized) periodogram
    # note pre-centering of y values!
    powers = lombscargle(x, rv - rv.mean(), ang_freqs)

    # normalize the power
    N = len(x)
    powers *= 2 / (N * rv.std() ** 2)
    return periods, powers

#slighty altered periodogram function, computes data window for a set of visits
def dataWindow(x, f, max_period):
    x = np.array(x)
#    delta_x = np.inf 
#    for i in range(0, len(x)-2):
#        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
#            delta_x = x[i+1]-x[i]
    periods = np.linspace(0.04167, max_period, num = f)

    # convert period range into frequency range
    ang_freqs = 2 * np.pi / periods

    # compute the (unnormalized) periodogram
    # note pre-centering of y values!
    powers = lombscargle(x, np.ones(len(x)), ang_freqs)

    # normalize the power
    N = len(x)
    powers *= 2 / N
    return periods, powers

#this function removes nan cells from the bad RV visits, and deletes the accompanying JD element 
#from a copy tied to the specific rv list
def adjustment(x, rv):
    newJD, newRV = np.array([]), np.array([])
    for i in range(len(np.where(np.isfinite(rv))[0])):
        newJD = np.append(newJD, x[np.where(np.isfinite(rv))[0][i]])
        newRV = np.append(newRV, rv[np.where(np.isfinite(rv))[0][i]])
    return newJD, newRV

#function converts measurements in time into measurements in orbital phase (from 0-1)
#function is only useful after T and P have been determined
def phases(P, times):
    phased_Times = np.array([])
    for i in range(len(times)):
        phased_Times = np.append(phased_Times, times[i]/P-int(times[i]/P))
        if phased_Times[i] < 0:
            phased_Times[i] = phased_Times[i]+1
    return phased_Times

#function calculates mass ratio and error of both the regression and the slope parameter
def massRatio(x, y, system):
    y = [datum[1] for datum in system if not np.isnan(datum[1]+datum[2])] #primary component
    x = [datum[2] for datum in system if not np.isnan(datum[1]+datum[2])] #secondary
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    R2 = r_value**2
    x_bar = np.asarray(x).mean()
    sx2 = ((x-x_bar)**2).sum()
    slope_err = std_err * np.sqrt(1./sx2)
    return -slope, intercept, R2, std_err, slope_err

#function finds local maxima above a specified cutoff power
def maxima(cutoff, x, y, y2):
    power = y*y2
    maxima = np.array([])
    for i in range(1, len(power)-2):
        if power[i-1] < power[i] and power[i] > power[i+1] and power[i] > cutoff:
            maxima = np.append(maxima, x[i])
    return maxima

#provides starting point for the MCMC
def initialGuess(lower, upper, JDp, RVp):
    return curve_fit(alteredRV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]

#provides starting point for the MCMC, with circular orbit
def initialGuessNoE(lower, upper, JDp, RVp):
    return curve_fit(alteredNoERV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]

#function calculates and returns the residuals of a particular fit w.r.t. the data
def residuals(JDp, JDs, mass_ratio, primary, secondary, parameters):
    r = np.sqrt(sum((np.asarray(primary)-RV(JDp, mass_ratio, parameters)[0])**2)
        +sum((np.asarray(secondary)-RV(JDs, mass_ratio, parameters)[1])**2))
    return r

def constraints(parameters, lower, upper):
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    if  lower[0] < K < upper[0] and lower[1] < e < upper[1] and lower[2] < w < upper[2] and lower[3] < T < upper[3] and lower[4] < P < upper[4] and lower[5] < y < upper[5]:
        return 0
    return -np.inf

def constraintsNoE(parameters, lower, upper):
    K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
    if  lower[0] < K < upper[0] and lower[1] < T < upper[1] and lower[2] < P < upper[2] and lower[3] < y < upper[3]:
        return 0
    return -np.inf
