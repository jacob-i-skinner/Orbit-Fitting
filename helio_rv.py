import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


pi      = np.pi
sin     = np.sin
cos     = np.cos
tan     = np.tan
arctan  = np.arctan
amax    = np.amax
sqrt    = np.sqrt

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
    primary : array_like[len(x)]
        The RV curve of the primary component.
    
    secondary : array_like[len(x)]
        The RV curve of the secondary component.
            
    '''
    if len(parameters) == 4: # Circular orbit case.
        K, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3]
        return K*cos((2*pi/P)*(x-T)) + y, (-K/q)*cos((2*pi/P)*(x-T)) + y

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
        if amax(E1-E0) < 1e-9:
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

#----------------------------------------------------------------------------------------------------#

mass_ratio, parameters = 0.7,[32.1675062656, 0.8, 1.1,
                                         2456260.80134, 2.6320929857, 70]
#----------------------------------------------------------------------------------------------------#

x = np.linspace(0, parameters[4], num=1000)
fig, ax = plt.figure(figsize=(15,8)), plt.subplot(111)
primary, secondary = RV(x, mass_ratio, parameters)
ax.plot(x, np.ones(len(x))*parameters[5], 'k' , lw = 2, label='Systemic Velocity')
ax.plot(x/parameters[4], primary, 'b', lw=2)
ax.plot(x/parameters[4], secondary, 'r', lw=2)
plt.xlabel('Orbital Phase', fontsize = 18)
plt.ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 18)
ax.set_xlim([0,1])
plt.show()