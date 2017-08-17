# Support functions for the fitting scripts in orbit-fitting.
import numpy as np

'''
Defining these functions here saves a bit of time each time
they are called because numpy does not need to be referenced.

It's relevant because some of these functions are called several
times for every call to RV, and RV is called twice for every calculation
of the likelihood, which is run a mimimum of nwalkers*nsteps times
during the random walk, which, as of the time of
this writing is 1-4 millions.
'''
pi      = np.pi
sin     = np.sin
cos     = np.cos
tan     = np.tan
arctan  = np.arctan
amax    = np.amax
sqrt    = np.sqrt
asarray = np.asarray
append  = np.append
median  = np.median
inf     = np.inf
insert  = np.insert

def massLimit(q, K, e, P):
    '''
    Compute the lower limit of the Primary mass.

    Parameters
    ----------
    q : float
        mass ratio
    
    K : float
        semi-amplitude
    
    e : float
        eccentricity
    
    P : float
        period
    
    Returns
    -------
    M : float
        An estimate of the lower limit of the primary
    component's mass, in terms of Solar masses.
    '''
    # Convert K and P into SI units.
    K, P = K*1000, P*24*3600

    # Build the parts of the equation
    A = 1/((1+q)*((1-0.25*e**2-0.46875*e**4)**3))
    B = (P*K**3)/(2*pi*6.67428e-11)

    M = A*B

    # Return M in terms of solar mass
    return round(M/1.989e30, 3)


def coverage(RVp, RVs):
    '''
    Calculate the velocity span covered by the data.
    (Equation 23, Troup, et. al. 2016)

    Parameters
    ----------
    RVp : list
        Primary radial velocity data.

    RVs : list
        Secondary radial velocity data.

    Returns
    -------
    V_coverage : float
        Value between 0 and 1.
    '''
    # Data must be sorted.
    RVp, RVs = sorted(RVp), sorted(RVs)

    N = len(RVp)

    # Create and populate the array of differences.
    prim_diff = np.empty(N-1)
    for i in range(0,N-1):
        prim_diff[i] = RVp[i+1] - RVp[i]

    # Sum the squares of the differences.
    prim_sum = sum(prim_diff**2)

    prim = 1 - prim_sum/(max(RVp)-min(RVp))**2

    prim_cov = (N/(N-1))*prim
    
    # Repeat this process for the secondary.
    N = len(RVs)
    
    sec_diff = np.empty(N-1)
    for i in range(0,N-1):
        sec_diff[i] = RVs[i+1] - RVs[i]
    
    sec_sum = sum(sec_diff**2)

    sec = 1 - sec_sum/(max(RVs)-min(RVs))**2

    sec_cov = (N/(N-1))*sec

    # Average and return the two values
    return 0.5 * (prim_cov+sec_cov)


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
        
    parameters : iterable[6]
        The set of orbital elements with which to generate the curves.
    
    Returns
    -------
    [primary, secondary] : [array_like[len(x)], array_like[len(x)]]
        The primary and secondary RVs for a given time or list of times, x. 
            
    '''
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    
    # M (mean anomaly) is a function of x (time).
    M   = (2*pi/P)*(x-T)
    
    # E1 (eccentric anomaly) is a function of M.
    E1  = M + e*sin(M) + (e**2)*sin(2*M)/2
    
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

    # Compute and return the final curves.
    return K*(cos(v+w) + (e*cos(w)))+y, (-K/q)*(cos(v+w) + (e*cos(w)))+y


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

    # If there is an element of RV marked to ignore, remove the
    # element and the same element from JD as well.
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

    # x%P gives how far into a given period x lies, dividing into P
    # normalizes the result, giving the orbital phase.
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
        Mass Ratio of the system. The ratio of the secondary
        component mass to the primary.
    
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
    # Structure is the same as that of RV. (not anymore, RV has been modified)
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
    
    # This is a simple harmonic function.
    def alteredNoERV(x, K, T, P, y):
        return K*cos((2*pi/P)*(x-T))+y

    return curve_fit(alteredNoERV, JDp, np.asarray(RVp), bounds=(lower, upper))[0]


def residuals(parameters, mass_ratio, RVp, RVs, JDp, JDs):
    '''
    Compute the square root of the N normalized sum of the squares
    of the observed - computed.
    
    Parameters
    ----------
    parameters : ndarray(ndim)
        The values of the orbital elements needed to generate the RV curve.

    mass_ratio : float
        Ratio of secondary mass to primary mass.
    
    RVp : list
        Primary observed velocities.

    RVp : list
        Secondary observed velocities.

    JDp : list
        Primary observation times.
    
    JDs : list
        Secondary observation times.
    
    Returns
    -------
    r : float
        Total residual error.
    '''

    # Find the sum of the squares of the differences between the observed
    # vaues and those computed from the curve.
    p_err = sum((asarray(RVp)-RV(JDp, mass_ratio, parameters)[0])**2)

    s_err = sum((asarray(RVs)-RV(JDs, mass_ratio, parameters)[1])**2)

    # We do not want the error to rise with the number of data points
    # so the result is divided into the number of data points.
    p_err, s_err = p_err/len(RVp), s_err/len(RVs)

    # Find the square root of the remaining sum.
    r = sqrt(p_err + s_err)

    return r


def uncertainties(parameters, q, RVp, RVs, JDp, JDs):
    '''
    TO DO : prevent infinite loops from poor fits.

    Find the distance along each axis at which error increases
    by an arbitrary(?) factor from the minimum.
    
    Parameters
    ----------
    parameters : ndarray(ndim)
        The values of the orbital elements needed to calculate error.

    q : float
        Ratio of secondary mass to primary mass.
    
    RVp : list
        Primary observed velocities.

    RVp : list
        Secondary observed velocities.

    JDp : list
        Primary observation times.
    
    JDs : list
        Secondary observation times.
    
    Returns
    -------
    values_and_uncertainties : ndarray(ndim, 3)
        Array containing the calculated values and uncertainties.
        example shape:
        value0 +uncertainty0 -uncertainty0
        value1 +uncertainty1 -uncertainty1
    '''
    
    if len(parameters) == 4:
        parameters = insert(parameters, 1, [0,0])
    
    # Calculate the error at the location of best fit.
    error = residuals(parameters, q, RVp, RVs, JDp, JDs)
    
    # Create arrays to store the upper and lower uncertainty values.
    high = np.empty(6)
    low  = np.empty(6)
    
    # We do not want to alter parameters, so we define position.
    position = [x for x in parameters]
    
    # Loop over each dimension.
    for i in range(6):
        
        # One loop for upper value, one for the lower.
        for j in range(6):
            
            # Initial step size, shrinks each time direction is changed.
            if j%2 == 0:
                step = 0.1
            else:
                step = -0.1
            
            position = [x for x in parameters]

            # Iteratively zero in on the value until specified precision is reached.
            while abs(step) > 1e-9:

                # Move position along given axis until value has been passed.
                while residuals(position, q, RVp, RVs, JDp, JDs) < error*1.2:
                    position[i] = position[i] + step
                
                # Once the value has been passed, shrink the step size.
                step = step/2

                # Start heading the other way.
                while residuals(position, q, RVp, RVs, JDp, JDs) > error*1.2:
                    position[i] = position[i] - step

                step = step/2
            
            # Save the values.
            if j%2 == 0:
                high[i] = position[i]    
            else:
                low[i] = position[i]
    
    # Rewrite the values as the differences from the best fit.
    low, high = parameters - low, high - parameters

    # Return a nicely shaped array with the zeros removed if it's circular.
    if parameters[1] == parameters[2]:
        return np.transpose(np.delete(np.array([parameters, high, low]), [1,2], axis=1))
    else:
        return np.transpose(np.array([parameters, high, low]))


def walkers(nsteps, ndim, cutoff, sampler):
    '''
    Create a plot showing the path of each walker in each
    dimension over the course of the random walk.

    Parameters
    ----------
    nsteps : int
        The number of steps that were taken during the random walk.
    
    ndim : int
        The number of dimensions explored during the random walk.
    
    cutoff : int
        Steps before the cutoff are ignored when creating the samples
        array, and so are ignored when calculating values, and when creating
        the corner plot.

    sampler : emcee object
        The object created by emcee during the random walk,
        it contains the information about which walker was where
        at each step.


    Returns
    -------
    fig : fig object
        Something to either plot or save.

    '''
    from matplotlib import pyplot as plt
    
    # A bit of time is saved by making this local 
    # and not referencing numpy everytime the loop iterates.
    linspace = np.linspace

    # Make the labels.
    label= ['$K$', '$e$', '$\omega$', '$T$', '$P$', '$\gamma$']
    if ndim == 4:
        del label[1:3]
    
    # Create the fig object.
    fig, ax = plt.subplots(ndim, 1, sharex='col')
    plt.xlabel('Step Number (Cutoff step: %s)'%(cutoff), fontsize = 18)
    ax[0].set_title('Walker Positions During Random Walk', fontsize = 18)
    fig.set_figheight(20)
    fig.set_figwidth(15)

    # Populate the i-th axis with lines showing each walker's location 
    # along that axis, as a function of the step number.
    for i in range(ndim):
        for j in range(len(sampler.chain[:, 0, i])):
            ax[i].plot(linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.05)
        ax[i].set_ylabel(label[i], rotation = 0, fontsize = 18)
        ax[i].yaxis.set_label_coords(-0.06, 0.5)
    
    return fig


def corner(ndim, samples, parameters):
    '''
    Create a plot showing the all of the samples (after the cutoff) drawn
    from the distribution. There are (ndim choose 2) + ndim subplots.
    ndim subplots show histograms of the samples along a single axis.
    The other (ndim choose 2) subplots are projections of the sample into
    various axial planes. This is useful for seeing covariances among parameters.

    Parameters
    ----------
    ndim : int
        Dimensionality of the sample, typically 6 or 4.

    samples : array(nsteps*nwalkers, ndim)
        A collection of coordinates resulting from the random walk.
        
    parameters : list
        Values to be displayed as the 'truth' values. They represent the
        coordinate of the position which maximizes the estimated PDF.
    
    Returns
    -------
    fig : fig object
        Something to either plot or save.

    '''
    import corner
    from matplotlib import pyplot as plt

    # Rearrange samples to make dynamic bounds setting more readable.
    samples_T = np.transpose(samples)

    # Set up bounds and labels for 4 or 6 dimensional cases.
    if ndim == 4:
        bounds = [[np.amin(samples_T[0]), np.amax(samples_T[0])],
                  [np.amin(samples_T[1]), np.amax(samples_T[1])],
                  [np.amin(samples_T[2]), np.amax(samples_T[2])],
                  [np.amin(samples_T[3]), np.amax(samples_T[3])]]
        labels = ["$K$", "$T$", "$P$", "$\gamma$"]
 
    elif ndim == 6:
        bounds = [[np.amin(samples_T[0]), np.amax(samples_T[0])],
                  [np.amin(samples_T[1]), np.amax(samples_T[1])],
                  [np.amin(samples_T[2]), np.amax(samples_T[2])],
                  [np.amin(samples_T[3]), np.amax(samples_T[3])],
                  [np.amin(samples_T[4]), np.amax(samples_T[4])],
                  [np.amin(samples_T[5]), np.amax(samples_T[5])]]
        labels = ["$K$", "$e$", "$\omega$", "$T$", "$P$", "$\gamma$"]

    # Create the figure.
    fig = corner.corner(samples, bins = 80, range = bounds, labels = labels,
                        smooth = 1.2,truths = parameters, show_titles = True,
                        quantiles = [0.16, 0.84], title_kwargs = {"fontsize": 18})
    return fig


def maximize(samples):
    '''
    Use Kernel Density Estimation to find a continuous PDF
    from the discrete sampling. Maximize that distribution.

    Parameters
    ----------
    samples : list (shape = (nsamples, ndim))
        Observations drawn from the distribution which is
        going to be fit.
    
    Returns
    -------
    maximum : list(ndim)
        Maxima of the probability distributions along each axis.
    '''
    from scipy.optimize import minimize
    from scipy.stats import gaussian_kde as kde

    # Give the samples array the proper shape.
    samples = np.transpose(samples)
    
    # Estimate the continous PDF.
    estimate = kde(samples)

    # Take the minimum of the estimate.
    def PDF(x):
        return -estimate(x)
    
    # Initial guess on maximum is made from 50th percentile of the sample.
    p0 = [np.percentile(samples[i], 50) for i in range(samples.shape[0])]

    # Calculate the maximum of the distribution.
    maximum = minimize(PDF, p0).x

    return maximum


def logLikelihood(guess, q, RVp, RVs, JDp, JDs, lower, upper):
    '''
    Calculate the likelihood of a set of orbital elements being
    the 'true' values. Values are negative approaching 0! emcee
    does its best to maximize this value.

    Parameters
    ----------
    guess : list
        The set of parameters to be checked. 6 for eccentric, 4 for circular.

    q : float
        Ratio of secondary mass to primary mass.
    
    RVp : list
        Primary observed velocities.

    RVp : list
        Secondary observed velocities.

    JDp : list
        Primary observation times.
    
    JDs : list
        Secondary observation times.
    
    lower : list
        Lower bound of allowed values.
    
    upper : list
        upper bound of allowed values.

    Returns
    -------
    -inf : nan
        Returned if any element of guess falls outside of bounds.

    log_like : float
        Natural log of a gaussian function with an argument equal to
        the sum of the squares of the observed - computed given the
        guess, for each observation in the dataset.

    '''
    JD_median = median(append(JDs, JDp))

    # If fit is assumed circular, fill in the missing e and w.
    if len(guess) == 4 :
        guess = insert(guess, 1, [0,0])
    
    K, e, w, T, P, y = guess[0], guess[1], guess[2], guess[3], guess[4], guess[5]

    '''
        Check for out-of-bounds values, if a bound condition is not met,
    the function exits with -inf. Separating the checks is messier,
    but it means that the function exits as soon as any one value lies
    out-of-bounds (faster), instead of checking ALL of them (slower).
    '''
    if lower[0] > K:
        return -inf
    if K > upper[0]:
        return -inf

    if lower[1] > e:
        return -inf
    if e > upper[1]:
        return -inf
        
    if lower[2] > w:
        return -inf
    if w > upper[2]:
        return -inf
    
    if JD_median-0.5*guess[4] > T:
        return -inf
    if T > JD_median+0.5*guess[4]:
        return -inf
    
    if lower[4] > P:
        return -inf
    if P > upper[4]:
        return -inf
    
    if lower[5] > y:
        return -inf
    if y > upper[5]:
        return -inf
    
    # log_like = the log-lilekihood, -1/2 * the sum of the squares of the primary
    # and secondary observed - computed.
    
    #log_like = -0.5 * (sum((asarray(RVp)-RV(JDp, q, guess)[0])**2) + sum((asarray(RVs)-RV(JDs, q, guess)[1])**2))
    return -residuals(guess, q, RVp, RVs, JDp, JDs)
    #return log_like
    

def MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower, upper, ndim, nwalkers, nsteps, threads):
    '''
    Use an affine-invariant ensemble sampler to probe the probability
    density function defined by the likelihood function and the dataset.
    likelihood and MCMC were adapted from the "fitting a model to data"
    example by Dan Foreman-Mackey, on the emcee website: http://dan.iel.fm/emcee/current/

    Parameters
    ----------
    mass_ratio : float
        Ratio of secondary mass to primary mass.
    
    RVp : list
        Primary observed velocities.

    RVp : list
        Secondary observed velocities.

    JDp : list
        Primary observation times.
    
    JDs : list
        Secondary observation times.
    
    lower : list
        Lower bound of allowed values.
    
    upper : list
        Upper bound of allowed values.
    
    ndim : int
        Number dimensions, 6 or 4.
    
    nwalkers : int
        Number of walkers in the ensemble.

    nsteps : int
        Number of steps of the random walk.

    threads : int
        Number of threads to run the walk over.
        Values other than 1 are not compatible with Windows OS.

    Returns
    -------
    sampler : emcee object
        The results of the random walk. Its atribute "chain"
        is an array of size (ndim, nwalkers, nsteps) and stores
        the paths of each walker over the walk.
        See http://dan.iel.fm/emcee/current/api/#emcee.EnsembleSampler
        for more details.
    '''
    import emcee

    # Create an array to store the walker positions.
    position = np.empty([ndim,nwalkers])
    
    # Evenly space the walkers in the allowed ranges.
    if ndim == 4:
        for i in range(ndim):
            position[i] = np.linspace(np.delete(lower, [1,2])[i], np.delete(upper, [1,2])[i], num=nwalkers)
    else:
        for i in range(ndim):
            position[i] = np.linspace(lower[i], upper[i], num=nwalkers)

    # Reshape position to play nice with emcee.
    position = np.transpose(position)

    # Create the sampler object.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logLikelihood, a=8.0,
                                    args=(mass_ratio, RVp, RVs, JDp, JDs, lower, upper), threads=threads)
    
    # Do the run.
    sampler.run_mcmc(position, nsteps)
    
    return sampler