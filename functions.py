import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lombscargle
from scipy import stats

def RV(x, mass_ratio, parameters): #function generates RV values plot from given parameters
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2) #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0 = E1
        M0 = E0 - e*np.sin(E0)
        E1 = E0 +(M-M0)/(1-e*np.cos(E0))
        if np.amax(E1-E0) < 1E-9:
            break
    nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p, s = (K*(np.cos(nu+w) + (e*np.cos(w)))+y), ((-K/mass_ratio)*(np.cos(nu+w) + (e*np.cos(w)))+y)
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

#function computes and returns mass ratio and intercept
#def massRatio():
#    p =[datum[1] for datum in system if not np.isnan(datum[1]+datum[2])]
#    s =[datum[2] for datum in system if not np.isnan(datum[1]+datum[2])]
#    m,b = np.polyfit(s, p, 1)
#    return -m, b

#this function removes nan cells from the bad RV visits, and deletes the accompanying JD element 
#from a copy tied to the specific rv list
def adjustment(x, rv):
    adjusted_x = np.asarray(x)
    for i in range(0, len(x)-1):
        if i == len(rv):
            break
        if np.isnan(rv[i]):
            rv         = np.delete(rv, i)
            adjusted_x = np.delete(adjusted_x, i)
    return adjusted_x, rv

#function converts measurements in time into measurements in orbital phase (from 0-1)
#function is only useful after T and P have been determined
def phases(P, T, times):
    phased_Times = np.array([])
    for i in range(0, len(times)):
        phased_Times = np.append(phased_Times, ((times[i])-T)/P-int(((times[i])-T)/P))
        if phased_Times[i] < 0:
            phased_Times[i] = phased_Times[i]+1
    return phased_Times

#function calculates mass ratio and error of both the regression and the slope parameter
def massRatio(x,y, system):
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