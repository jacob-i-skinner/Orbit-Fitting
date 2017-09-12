import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
rcParams.update({'figure.autolayout' : True})

RV = f.RV
phases = f.phases
adjustment = f.adjustment
residuals = f.residuals
mass_ratio = 0.844

samples = np.genfromtxt('data/1720+4205/1720+4205.tbl 50.222 error samples.txt', usecols=(4), skip_header=0, delimiter=',')

'''
def maximize(samples, dim):
     ''''''
     Use Kernel Density Estimation to find a continuous PDF
     from the discrete sampling. Maximize that distribution.
 
     Parameters
     ----------
     samples : list (shape = (nsamples, ndim))
         Observations drawn from the distribution which is
         going to be fit.

    dim : integer
        The axis of the sample to minimize.
     
     Returns
     -------
     maximum : list(ndim)
         Maxima of the probability distributions along each axis.
     ''''''
     from scipy.optimize import minimize
     from scipy.stats import gaussian_kde as kde
 
     # Give the samples array the proper shape.
     samples = np.transpose(samples)
     
     # Estimate the continous PDF.
     estimate = kde(samples[dim])

     # Initial guess on maximum is made from 50th percentile of the sample.
     p0 = np.percentile(samples[dim], 50)
 
     # Calculate the maximum of the distribution.
     minimum = minimize(estimate, p0).x
 
     return minimum



# Fold e, w, and T if e is negative.
P = np.percentile(np.transpose(samples)[4], 50)
T_minimum = maximize(samples, 3)
w_minimum = maximize(samples, 2)

for i in range(4000000):
    samples[i][3] = samples[i][3] - 2456000
    if samples[i][1] < 0:
        samples[i][1] = -samples[i][1]
        samples[i][2] = samples[i][2] - np.pi
        samples[i][3] = samples[i][3] - P/2
      

for i in range(4000000):
    samples[i][1] = samples[i][1] - 2456000
    #if samples[i][1] < 0:
    #    samples[i][1] = -samples[i][1]
    #    samples[i][2] = samples[i][2] - np.pi
    #    samples[i][3] = samples[i][3] - P/2
'''

#parms = [x for x in np.transpose(np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                               zip(*np.percentile(samples, [16, 50, 84], axis=0))))))[0]]

file     = 'data/1720+4205/1720+4205.tbl'

fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.hist(samples, bins='auto', normed=True, color='black')
ax.set_title(1720+4205)
ax.set_ylabel('Frequency')
ax.set_xlabel('Period (days)')
ax.set_xlim(1,10)
plt.savefig(file + ' histogram.eps')
plt.show()

'''
def corner(ndim, samples, parameters):
    ''''''
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
    ''''''
    import corner
    from matplotlib import pyplot as plt

    # Rearrange samples to make dynamic bounds setting more readable.
    samples_T = np.transpose(samples)

    # Set up bounds and labels for 4 or 6 dimensional cases.
    if ndim == 4:
        labels = ["$K$", "$T$", "$P$", "$\gamma$"]
 
    elif ndim == 6:
        labels = ["$K$", "$e$", "$\omega$", "$T$", "$P$", "$\gamma$"]

    # Create the figure.
    fig = corner.corner(samples, bins = 80, labels = labels,
                        smooth = 1.2, show_titles = True, truths=parameters,
                        quantiles = [0.16, 0.84], title_kwargs = {"fontsize": 18})
    return fig

corner(4, samples, parms).savefig(file + ' 6 dimension corner plot.eps')


data       = np.genfromtxt(file, skip_header=1, usecols=(1,2,3))

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[2] for datum in data]
JDp, JDs        = JD, JD

JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#create the curves plot
fig = plt.figure(figsize=(11,10))
gs = GridSpec(2,1, height_ratios = [4,1])
ax1 = fig.add_subplot(gs[0,0])
ax1.tick_params(labelsize=14)
ax2 = fig.add_subplot(gs[1,0])
ax2.tick_params(labelsize=14)
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
#fig.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)

x = np.linspace(0, parms[-2], num=1000)
primary, secondary = RV(x, mass_ratio, parms)

ax1.plot(x, np.ones(len(x))*parms[-1], 'k', lw=1 , label='Systemic Velocity')
ax1.plot(x/parms[-2], primary, 'b', lw=1, label='Primary Curve')
ax1.plot(x/parms[-2], secondary, 'r', lw=1, label='Secondary Curve')

ax1.plot(phases(parms[-2], JDp), RVp, 'k.', label='Primary RV Data') #data phased to result period
ax1.plot(phases(parms[-2], JDs), RVs, 'k.', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot((0, 1), np.zeros(2), 'k', lw = 1)
ax2.plot(phases(parms[-2], JDp), RVp-RV(JDp, mass_ratio, parms)[0], 'bo')
ax2.plot(phases(parms[-2], JDs), RVs-RV(JDs, mass_ratio, parms)[1], 'ro')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 20)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 20)
ax2.set_ylabel('O - C', fontsize = 20)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
#plt.savefig(file + ' curve results.eps')

print(round(residuals(parms, mass_ratio, RVp, RVs, JDp, JDs), 3))

plt.show()
'''