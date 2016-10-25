import numpy as np
import emcee
from scipy import stats
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

#returns RV values over a time interval, of a curve from given parameters
#x is an array-like representing samples from a time interval, mass_ratio is a float, parameters is a 6 element list
def RV(x, mass_ratio, parameters):
    #if orbit is assumed circular (4 elements passed) add zeroes for e and omega values
    #into parameter list
    if len(parameters) == 4:
        parameters = list(parameters)
        parameters.insert(1, 0), parameters.insert(1, 0)
    check = 1 #this variable is used to prevent the while loop from continuing infinitely,
              #if the error in the has reached a lower limit above the specified 1e-9, the check provides an escape from the loop
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2) #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0    = E1
        M0    = E0 - e*np.sin(E0)
        E1    = E0 +(M-M0)/(1-e*np.cos(E0))
        #this loop is 
        if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p, s = (K*(np.cos(nu+w) + (e*np.cos(w)))+y), ((-K/mass_ratio)*(np.cos(nu+w) + (e*np.cos(w)))+y) 
    return p, s

#This periodogram function was adapted from Jake Vanderplas' article "Fast Lomb-Scargle Periodograms in Python"
def periodogram(x, rv, f, max_period):
    x = np.array(x)
    rv = np.array(rv)
    #currently using a pseudo-nyquist lower limit in favor of an arbritrary lower limit, to avoid spurious periods at high frequencies
    delta_x = np.inf
    for i in range(0, len(x)-2):
        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
            delta_x = x[i+1]-x[i]
    # lower limit of periods range set to one hour
    periods = np.linspace(delta_x, max_period, num = f) #f here is the number of samples taken over the range of periods
    #one hour limit - 0.04167

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
    #pseudo-nyquist lower limit
    delta_x = np.inf 
    for i in range(0, len(x)-2):
        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
            delta_x = x[i+1]-x[i]
    periods = np.linspace(delta_x, max_period, num = f)
    #one hour limit - 0.04167

    # convert period range into frequency range
    ang_freqs = 2 * np.pi / periods

    # compute the (unnormalized) periodogram
    # y values are all 1
    powers = lombscargle(x, np.ones(len(x)), ang_freqs)

    # normalize the power
    N = len(x)
    powers *= 2 / N #standard deviation is set to 1
    return periods, powers

#function removes nan cells from the bad RV visits, and deletes the accompanying JD element 
#returns adjusted RV and JD lists
def adjustment(x, rv):
    newJD, newRV = np.array([]), np.array([])
    for i in range(len(np.where(np.isfinite(rv))[0])):
        newJD = np.append(newJD, x[np.where(np.isfinite(rv))[0][i]])
        newRV = np.append(newRV, rv[np.where(np.isfinite(rv))[0][i]])
    return newJD, newRV

#converts measurements in time into measurements in orbital phase (from 0-1)
#function is only useful after T and P have been determined
def phases(P, times):
    phased_Times = np.array([])
    for i in range(len(times)):
        phased_Times = np.append(phased_Times, times[i]/P-int(times[i]/P))
        if phased_Times[i] < 0:
            phased_Times[i] = phased_Times[i]+1
    return phased_Times

#returns mass ratio, and another of values related to the linear fit
def massRatio(x, y, system):
    y = [datum[1] for datum in system if not np.isnan(datum[1]+datum[2])] #primary component
    x = [datum[2] for datum in system if not np.isnan(datum[1]+datum[2])] #secondary
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    R2 = r_value**2
    x_bar = np.asarray(x).mean()
    sx2 = ((x-x_bar)**2).sum()
    slope_err = std_err * np.sqrt(1./sx2)
    return -slope, intercept, R2, std_err, slope_err

#returns local maxima above a specified cutoff power
def maxima(cutoff, x, y):
    maxima = np.array([])
    for i in range(1, len(y)-2):
        if y[i-1] < y[i] and y[i] > y[i+1] and y[i] > cutoff:
            maxima = np.append(maxima, x[i])
    return maxima

#provides starting point for the MCMC, returns parameters (within the specified bounds) from a simple but fast fit
def initialGuess(lower, upper, JDp, RVp):
    #initialGuess is a simple fitter, so alteredRV exists to return just the primary curve
    def alteredRV(x, K, e, w, T, P, y): #different argument structure accomodates scipy.curve_fit
        check = 1
        M = (2*np.pi/P)*(x-T)
        E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2)
        while True:
            E0 = E1
            M0 = E0 - e*np.sin(E0)
            E1 = E0 +(M-M0)/(1-e*np.cos(E0))
            if np.amax(E1-E0) < 1e-9 or check-np.amax(E1-E0) == 0:
                break
            else:
                check = np.amax(E1-E0)
        nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2))
        p = ((K)*(np.cos(nu+w) + (e*np.cos(w)))+y)
        return p
    return curve_fit(alteredRV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]

#same as initialGuess, but for the special case of a circular orbit.
def initialGuessNoE(lower, upper, JDp, RVp):
    def alteredNoERV(x, K, T, P, y):
        return K*np.cos((2*np.pi*x/P)+T)+y
    return curve_fit(alteredNoERV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]

#returns the residual error of the data w.r.t. a particular fit
#is now a properly "normalized" RMS error
def residuals(parameters, mass_ratio, RVp, RVs, JDp, JDs):
    r = np.sqrt(sum((np.asarray(RVp)-RV(JDp, mass_ratio, parameters)[0])**2)/(len(RVp))
        +sum((np.asarray(RVs)-RV(JDs, mass_ratio, parameters)[1])**2)/(len(RVs)))
    return r

#returns the coefficient of determination for a particular fit
def rSquared(parameters, mass_ratio, RVp, RVs, JDp, JDs):
    SSres = sum((np.asarray(RVp)-RV(JDp, mass_ratio, parameters)[0])**2)+sum((np.asarray(RVs)-RV(JDs, mass_ratio, parameters)[1])**2)
    SStot = sum((np.asarray(RVp)-np.mean(np.asarray(RVp)))**2)+sum((np.asarray(RVs)-np.mean(np.asarray(RVs)))**2)
    r2    = 1-SSres/SStot
    return r2

#the functions below are either the MCMC itself, or critical support functions
#they were adapted from the "fitting a model to data" example by Dan Foreman-Mackey, on the emcee website

def constraints(parameters, lower, upper):
    if len(parameters) == 4:
        K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
        if  lower[0] < K < upper[0] and lower[1] < T < upper[1] and lower[2] < P < upper[2] and lower[3] < y < upper[3]:
            return 0
        return -np.inf
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    if  lower[0] < K < upper[0] and -1 < e < 1 and -2*np.pi < w < 2*np.pi and lower[3] < T < upper[3] and lower[4] < P < upper[4] and lower[5] < y < upper[5]:
        return 0
    return -np.inf

#not technically probability, returns the negative infinity if parameters lie outside contraints, otherwise
#returns negative of RMS error, emcee tries to maximize this quantity
def probability(initial_guess, mass_ratio, RVp, RVs, JDp, JDs, lower, upper): #lnprob
    con = constraints(initial_guess, lower, upper)
    if not np.isfinite(con):
        return -np.inf
    return -residuals(initial_guess, mass_ratio, RVp, RVs, JDp, JDs)

def MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, ndim, nwalkers, nsteps, cores):
    #if the fit is circular...
    if ndim == 4:
        del lower_bounds[1:3], upper_bounds[1:3]
        initial_guess = initialGuessNoE(lower_bounds, upper_bounds, JDp, RVp)
        #initialize walkers 
        position = [initial_guess + 0.1*np.random.randn(ndim) for i in range(nwalkers)]
        #walkers distributed in gaussian ball around most likely parameter values
        #coefficients on random samples are proportional to "spread" of values
        for i in range(nwalkers):
            position[i][0] = initial_guess[0] + 2.5*np.random.randn(1) #K
            position[i][1] = initial_guess[1] +     np.random.randn(1) #T
            position[i][2] = initial_guess[2] + 2  *np.random.randn(1) #P
            position[i][3] = initial_guess[3] + 3  *np.random.randn(1) #y

        #create the sampler object and take a walk
        sampler = emcee.EnsembleSampler(nwalkers, ndim, probability,
                                        args=(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds), threads=cores)
        sampler.run_mcmc(position, nsteps)
        return sampler

    #otherwise, eccentric fit
    initial_guess = initialGuess(lower_bounds, upper_bounds, JDp, RVp)
    position = [initial_guess + 0.1*np.random.randn(ndim) for i in range(nwalkers)]
    for i in range(nwalkers):
        position[i][0] = initial_guess[0] + 5  *np.random.randn(1) #K
        position[i][1] = initial_guess[1] + 0.1*np.random.randn(1) #e
        position[i][2] = initial_guess[2] + 1  *np.random.randn(1) #w
        position[i][3] = initial_guess[3] +     np.random.randn(1) #T
        position[i][4] = initial_guess[4] + 2  *np.random.randn(1) #P
        position[i][5] = initial_guess[5] + 3  *np.random.randn(1) #y
    sampler = emcee.EnsembleSampler(nwalkers, ndim, probability,
                                    args=(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds), threads=cores)
    sampler.run_mcmc(position, nsteps)
    return sampler