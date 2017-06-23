# Support functions for the fitting scripts in orbit-fitting.

import numpy as np

pi      = np.pi
sin     = np.sin
cos     = np.cos
tan     = np.tan
arctan  = np.arctan
amax    = np.amax
sqrt    = np.sqrt
asarray = np.asarray


def RV(x, q, parameters):
    '''
    Computes radial velocity curves from given parameters, akin
    to defining mathematical function RV(x).
    This function is based on HELIO_RV from NASA's IDL library.
    
    Parameters
    ----------
    x : array_like
        Time-like variable. Because data are plotted with the curves after
        being phased into a single period, we care about the regime
        from x = T to x = T + P. x should have sufficient length to provide
        good resolution to the curve, and a range from 0 to at least P.

    q : float
        The ratio of the mass of the secondary star to the primary, or mass ratio.
        Conventionally this is smaller than one. q scales the amplitude
        of the less massive star.
        
    parameters : iterable[6 (or 4)]
        The set of orbital elements with which to generate the curves. length is
        6 for an eccentric orbit, 4 for a perfectly circular one.
    
    Returns
    -------
    [primary, secondary] : [array_like[len(x)], array_like[len(x)]]
        The primary and secondary RVs for a given time or list of times, x. 
            
    '''
    if len(parameters) == 4: # Circular orbit case.
        K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
        return (K*cos((2*pi/P)*(x-T))+y), ((-K/q)*cos((2*pi/P)*(x-T))+y)
    
    # Otherwise, give the full eccentric treatment.
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    # M (mean anomaly) is a function of x (time).
    M   = (2*pi/P)*(x-T)
    # E1 (eccentric anomaly) is a function of M.
    E1  = M + e*sin(M) + ((e**2)*sin(2*M)/2)
    
    step  = 0
    check = [1, 1]
    # Iteratively refine estimate of E1 from initial estimate.
    while True:
        E0  = E1
        M0  = E0 - e*sin(E0)
        E1  = E0 + (M-M0)/(1-e*cos(E0))
        
        # If desired or maximal precision is reached, break.
        # Usually this statement is enough to exit the loop.
        if amax(E1-E0) < 1e-8:
            break
        
        # If precision has maximized (error not shrinking), break.
        if check[0]-amax(E1-E0) == 0 or check[1]-amax(E1-E0) == 0:
            break
        
        # Keep track of the last 2 error values.
        check[step%2] = amax(E1-E0)
        step += 1
        

    # v (true anomaly) is a function of E1.
    v  = 2*arctan(sqrt((1 + e)/(1 - e))*tan(E1/2))

    # Compute and return the final curves
    return (K*(cos(v+w) + (e*cos(w)))+y), ((-K/q)*(cos(v+w) + (e*cos(w)))+y)


def periodogram(x, rv, f, max_period):
    '''
    Computes a Lomb-Scargle Periodogram of the input RV data.
    This was adapted from Jake Vanderplas' article "Fast Lomb-Scargle Periodograms in Python."
    
    Parameters
    ----------
    x : list
        The times at which the data points were gathered.
    
    rv : list
        The values of the measurand at the corresponding time, x.

    f : float
        The number of samples to take over the interval.
        
    max_period : float
        The maximum of the interval of periods to check.
    
    Returns
    -------
    periods : array_like[len(f)]
        Equally spaced array of possible period values.

    powers : array_like[len(f)]
        The calculated Power values over the range of periods,
        these form the normalized Lomb-Scargle Periodogram.

    delta_x : float
        The smallest separation between two values in x.

    '''
    from scipy.signal import lombscargle
    # Sort the time data chronologically.
    x = np.sort(np.array(x))
    rv = np.array(rv)
    # Start delta_x very large
    delta_x = np.inf
    
    # Iteratively lower delta_x
    for i in range(0, len(x)-2):        
        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
            delta_x = x[i+1]-x[i]

    # Compute the periodogram
    periods = np.linspace(delta_x, max_period, num = f)
    ang_freqs = 2 * pi / periods
    powers = lombscargle(x, rv - rv.mean(), ang_freqs)
    powers *= 2 / (len(x) * rv.std() ** 2)

    return periods, powers, delta_x


def dataWindow(x, f, max_period):
    '''
    Computes a data window of the dataset. That is, a periodogram with
    the all of the RV values and variances set to 1.
    
    Parameters
    ----------
    x : list
        The times at which the data points were gathered.
    
    f : float
        The number of samples to take over the interval.
        
    max_period : float
        The maximum of the interval of periods to check.

    Returns
    -------
    periods : array_like[len(f)]
        Equally spaced array of possible period values.

    powers : array_like[len(f)]
        The calculated Power values over the range of periods,
        these form the normalized Lomb-Scargle Periodogram.

    '''
    from scipy.signal import lombscargle
    # Sort the time data chronologically.
    x = np.sort(np.array(x))
    delta_x = np.inf

    # Iteratively lower the measure delta_x
    for i in range(0, len(x)-2):
        if x[i+1]-x[i] < delta_x and x[i+1]-x[i] != 0:
            delta_x = x[i+1]-x[i]
    
    periods = np.linspace(delta_x, max_period, num = f)
    ang_freqs = 2 * pi / periods
    powers = lombscargle(x, np.ones(len(x)), ang_freqs)
    powers *= 2 / len(x)

    return periods, powers


def adjustment(x, rv):
    '''
    Data conditioner to remove bad data values.

    Parameters
    ----------
    x : list
        Times of observation.

    rv : list
        Observed radial velocities.
    
    Returns
    -------
    newJD : list
        Adjusted times. Times with bad RVs have been removed.

    newRV : list
        List of observations with bad RVs removed.
    
    '''
    newJD, newRV = np.array([]), np.array([])

    # If there is a of RV marked, remove the element and the same
    # element from JD as well.
    for i in range(len(np.where(np.isfinite(rv))[0])):
        newJD = np.append(newJD, x[np.where(np.isfinite(rv))[0][i]])
        newRV = np.append(newRV, rv[np.where(np.isfinite(rv))[0][i]])
    
    return newJD, newRV


def phases(P, times):
    '''
    Turns a list of times into a list of orbital phases with respect to P

    Parameters
    ----------
    P : float
        Period to which data are phased.

    times : list
        Times of observation.
    
    Returns
    -------
    phased_Times : list
        Times of observation in units of orbital phase.
    
    '''

    return [(x%P)/P for x in times]


def wilson(data):
    '''
    Calculate useful things like mass ratio and systemic velocity.

    Parameters
    ----------
    data : list
        Radial velocity pairs in a 2D list.

    Returns
    -------
    -slope : float
        Mass Ratio of the system.
    
    intercept : float
        y-intercept of the line which fits data.

    stderr : float
        Standard error of the estimated gradient.

    '''
    from scipy.stats import linregress
    
    # Primary RVs on y.
    y = [datum[1] for datum in data if not np.isnan(datum[1]+datum[2])]

    # Secondary RVs on x.
    x = [datum[2] for datum in data if not np.isnan(datum[1]+datum[2])]

    slope, intercept, rvalue, pvalue, stderr = linregress(x,y)
    
    return -slope, intercept, stderr



def initialGuess(lower, upper, JDp, RVp):
    '''
    Make a guess at the orbital element values.

    Parameters
    ----------
    lower : list(ndim)
        Lower bounds of the orbital elements.
    
    upper : list(ndim)
        Upper bounds of the orbital elements.
    
    JDp : list
        Times of observation of the primary component.
    
    RVp : list
        Radial velocities of the primary component.

    Returns
    -------
    (K, e, w, T, P, y) : list
        Output from curve_fit.
        
    '''
    from scipy.optimize import curve_fit
    
    # Slightly altered version of RV, to accomodate curve_fit.
    # Structure is the same as that of RV.
    def alteredRV(x, K, e, w, T, P, y):
        check = 1
        M = (2*pi/P)*(x-T)
        E1 = M + e*sin(M) + ((e**2)*sin(2*M)/2)
        while True:
            E0 = E1
            M0 = E0 - e*sin(E0)
            E1 = E0 +(M-M0)/(1-e*cos(E0))
            if amax(E1-E0) < 1e-9 or check-amax(E1-E0) == 0:
                break
            else:
                check = amax(E1-E0)
        nu = 2*arctan(np.sqrt((1 + e)/(1 - e))*tan(E1/2))
        p = ((K)*(cos(nu+w) + (e*cos(w)))+y)
        return p

    return curve_fit(alteredRV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]


def initialGuessNoE(lower, upper, JDp, RVp):
    '''
    Make a guess of the values of the elements, assuming it is circular.

    Parameters
    ----------
    lower : list(ndim)
        Lower bounds of the orbital elements.
    
    upper : list(ndim)
        Upper bounds of the orbital elements.
    
    JDp : list
        Times of observation of the primary component.
    
    RVp : list
        Radial velocities of the primary component.

    Returns
    -------
    (K, T, P, y) : list
        Output from curve_fit.

    '''
    from scipy.optimize import curve_fit
    
    # This is a simple trig function.
    def alteredNoERV(x, K, T, P, y):
        return K*cos((2*pi*x/P)+T)+y

    return curve_fit(alteredNoERV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]


def residuals(parameters, mass_ratio, RVp, RVs, JDp, JDs):
    r = sqrt( sum((asarray(RVp)-RV(JDp, mass_ratio, parameters)[0])**2)/(len(RVp))
             +sum((asarray(RVs)-RV(JDs, mass_ratio, parameters)[1])**2)/(len(RVs)))
    return r

''' Not used
#returns the coefficient of determination for a particular fit
def rSquared(parameters, mass_ratio, RVp, RVs, JDp, JDs):
    SSres = sum((np.asarray(RVp)-RV(JDp, mass_ratio, parameters)[0])**2)+sum((np.asarray(RVs)-RV(JDs, mass_ratio, parameters)[1])**2)
    SStot = sum((np.asarray(RVp)-np.mean(np.asarray(RVp)))**2)+sum((np.asarray(RVs)-np.mean(np.asarray(RVs)))**2)
    r2    = 1-SSres/SStot
    return r2
'''

#the functions below are either the MCMC itself, or critical support functions
#they were adapted from the "fitting a model to data" example by Dan Foreman-Mackey, on the emcee website

#create the walkers plot
def walkers(file, nsteps, ndim, sampler, results):
    from matplotlib import pyplot as plt
    linspace = np.linspace
    ones = np.ones
    label= ['$K$', '$e$', '$\omega$', '$T$', '$P$', '$\gamma$']
    if ndim == 4:
        del label[1:3]
    fig, ax = plt.subplots(ndim, 1, sharex='col')
    for i in range(ndim):
        for j in range(len(sampler.chain[:, 0, i])):
            ax[i].plot(linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.2)
        ax[i].plot(linspace(0, nsteps, num=nsteps) , np.ones(nsteps)*results[i][0], 'b', lw=2)
        ax[i].set_ylabel(label[i], rotation = 0, fontsize = 18)
        ax[i].yaxis.set_label_coords(-0.06, 0.5)
    plt.xlabel('Step Number', fontsize = 18)
    ax[0].set_title('Walker Positions During Random Walk', fontsize = 18)
    fig.set_figheight(20)
    fig.set_figwidth(15)
    plt.savefig(file + ' %s dimension walk results.png'%(ndim))
    return

def corner(file, ndim, samples, lower_bounds, upper_bounds, parameters):
    import corner
    from matplotlib import pyplot as plt
    truths = parameters
    samples_T = np.transpose(samples)
    if ndim == 4:
        bounds, labels = [[np.amin(samples_T[0]), np.amax(samples_T[0])],
                          [np.amin(samples_T[1]), np.amax(samples_T[1])],
                          [lower_bounds[2],             upper_bounds[2]],
                          [np.amin(samples_T[3]), np.amax(samples_T[3])]], ["$K$", "$T$", "$P$", "$\gamma$"]
 
    elif ndim == 6:
        bounds, labels = [[np.amin(samples_T[0]), np.amax(samples_T[0])],
                           [np.amin(samples_T[1]), np.amax(samples_T[1])],
                           [np.amin(samples_T[2]), np.amax(samples_T[2])],
                           [np.amin(samples_T[3]), np.amax(samples_T[3])],
                           [lower_bounds[4],             upper_bounds[4]],
                           [np.amin(samples_T[5]), np.amax(samples_T[5])]], ["$K$", "$e$", "$\omega$", "$T$", "$P$", "$\gamma$"]

    fig = corner.corner(samples, bins = 60, range = bounds, labels = labels, smooth = 0.8,
                        truths = truths,
                        quantiles=[0.16, 0.84], show_titles = False, title_kwargs = {"fontsize": 18})
    plt.savefig(file + ' %s dimension parameter results.png'%(ndim))
    return

'''
def constraints(parameters, lower, upper):
    if len(parameters) == 4:
        K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
        if  lower[0] < K < upper[0] and lower[1] < T < upper[1] and lower[2] < P < upper[2] and lower[3] < y < upper[3]:
            return 0
        return -np.inf
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    if  lower[0] < K < upper[0] and -1 < e < 1 and -2*pi < w < 2*pi and lower[3] < T < upper[3] and lower[4] < P < upper[4] and lower[5] < y < upper[5]:
        return 0
    return -np.inf
'''

def kernelDensityP(samples):
    '''
    Use Kernel Density Estimation to find the most probable
    values implied by the sample.

    Parameters
    ----------
    samples : list (shape = (nsamples, ndim))
        Observations drawn from the distribution which is
        going to be fit.
    
    Returns
    -------
    maximum : list(ndim)
        Maxima of the probability distributions along each dimensional axis.
    '''
    from scipy.optimize import minimize
    from scipy.stats import gaussian_kde as kde

    # Give the samples array the proper shape.
    samples = np.transpose(samples)
    

    # Define the esitmate of the sample.
    estimate = kde(samples)

    # Take the minimum of the estimate.
    def pdf(x):
        return -estimate(x)
    
    # Initial guess on maximum is made from 50th percentile of the sample.
    p0 = [np.percentile(samples[i], 50) for i in range(samples.shape[0])]

    # Calculate the maximum of the distribution.
    maximum = minimize(pdf, p0).x

    return maximum

#not technically probability, returns the negative infinity if parameters lie outside contraints, otherwise
#returns negative of RMS error, emcee tries to maximize this quantity

append  = np.append
median  = np.median
inf     = np.inf
def probability(guess, mass_ratio, RVp, RVs, JDp, JDs, lower, upper, nsteps, nwalkers): #lnprob
    JD_median = median(append(JDs, JDp))
    if len(guess) == 4 :
        K, T, P, y = guess[0], guess[1], guess[2], guess[3]
        if not (lower[0] < K < upper[0] and JD_median-0.5*guess[2] < T < JD_median+0.5*guess[2] and lower[2] < P < upper[2] and lower[3] < y < upper[3]):
        #if not (lower[0] < K < upper[0] and lower[1] < T < upper[1] and lower[2] < P < upper[2] and lower[3] < y < upper[3]):
            return -inf
        return -residuals(guess, mass_ratio, RVp, RVs, JDp, JDs)
    K, e, w, T, P, y = guess[0], guess[1], guess[2], guess[3], guess[4], guess[5]
    if not (lower[0] < K < upper[0] and lower[1] < e < upper[1] and lower[2] < w < upper[2] and JD_median-0.5*guess[4] < T < JD_median+0.5*guess[4] and lower[4] < P < upper[4] and lower[5] < y < upper[5]):
        return -inf
    return -residuals(guess, mass_ratio, RVp, RVs, JDp, JDs)

#This function is used to aid the fitter while it is doing the 1 dimensional T fit
def goodnessOfFit(T, parameters, mass_ratio, RVp, RVs, JDp, JDs, lower, upper): #lnprob
    JD_median = median(append(JDs, JDp))
    if not JD_median-0.5*parameters[4] < T < JD_median+0.5*parameters[4]:
        return -inf
    return -residuals([parameters[0], parameters[1], parameters[2], T, parameters[4], parameters[5]], mass_ratio, RVp, RVs, JDp, JDs)


def MCMC(mass_ratio, gamma, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, ndim, nwalkers, nsteps, cores):
    import emcee
    random = np.random.randn
    #deprecated by lowEFit, usually
    #if the fit is assumed to be circular, then ndim = 4, proceed accordingly
    if ndim == 4:
        del lower_bounds[1:3], upper_bounds[1:3]
        initial_guess = initialGuessNoE(lower_bounds, upper_bounds, JDp, RVp)
        #initialize walkers 
        position = [initial_guess + 0.1*random(ndim) for i in range(nwalkers)]
        #walkers distributed in gaussian ball around most likely parameter values
        #coefficients on random samples are proportional to "spread" of values
        for i in range(nwalkers):
            position[i][0] = initial_guess[0] + 4  *random(1) #K
            position[i][1] = initial_guess[1] +     random(1) #T
            position[i][2] = initial_guess[2] +     random(1) #P
            position[i][3] = gamma            + 3  *random(1) #y

        #create the sampler object and take a walk
        sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, a=4.0,
                                        args=(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, nsteps, nwalkers), threads=cores)
        sampler.run_mcmc(position, nsteps)
        return sampler

    #otherwise, eccentric fit
    initial_guess = initialGuess(lower_bounds, upper_bounds, JDp, RVp)
    position = [initial_guess + 0.1*random(ndim) for i in range(nwalkers)]
    #distribute the walkers around the values given by the initial guesser, in a 'gaussian ball'
    for i in range(nwalkers):
        position[i][0] = initial_guess[0] + 4  *random(1) #K
        position[i][1] = initial_guess[1] +0.05*random(1) #e
        position[i][2] = initial_guess[2] +     random(1) #w
        position[i][3] = initial_guess[3] +     random(1) #T
        position[i][4] = initial_guess[4] +     random(1) #P
        position[i][5] = gamma            + 3  *random(1) #y
    sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, a=2.0,
                                    args=(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, nsteps, nwalkers), threads=cores)
    sampler.run_mcmc(position, nsteps)
    return sampler

def lowEFit(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, nwalkers, nsteps, cores, parameters):
    import emcee
    initial_results = parameters
    random = np.random.randn
    position = [initial_results[3] + random(1) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, 1, goodnessOfFit, a=4.0,
                                    args=(parameters, mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds), threads=cores)
    sampler.run_mcmc(position, nsteps)
    return sampler