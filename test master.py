#import-libraries-and-data---------------------------------------------------------------------------------------#
import time
t0 = time.time()
import corner
import numpy as np
import functions as f
from scipy import stats
from matplotlib import pyplot as plt
filename     = 'Systems/2M03450783+3102335/4586_2M03450783+3102335_rvs.tbl'
system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
JDp, JDs        = JD, JD
samples         = 1000
max_period      = 9
power_cutoff    = 0.7
nwalkers, nsteps= 14, 100

#define-functions------------------------------------------------------------------------------------------------#

periodogram     = f.periodogram
dataWindow      = f.dataWindow
maxima          = f.maxima
phases          = f.phases
massRatio       = f.massRatio
adjustment      = f.adjustment
RV              = f.RV
residuals       = f.residuals
constraints     = f.constraints
MCMC            = f.MCMC

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilkinson plot (mass ratio)
mass_ratio, intercept, r_squared, standard_error, slope_error = massRatio(RVs,RVp, system)
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(RVs, RVp, 'k.')
x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]),-mass_ratio*np.array([np.nanmin(RVs), 
                                                                        np.nanmax(RVs)])+intercept
ax.plot(x, y)
ax.set_ylabel('Primary Velocity', size='15')
ax.set_xlabel('Secondary Velocity', size='15')
plt.savefig(filename + ' mass ratio.png')
print('mass ratio is ', mass_ratio, "+/-", standard_error)

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#calculate periodograms
x, y  = periodogram(JDp, RVp, samples, max_period)
y2    = periodogram(JDs, RVs, samples, max_period)[1]
y3,y4 = dataWindow(JDp, samples, max_period)[1], dataWindow(JDs, samples, max_period)[1]

#plot periodograms
fig, ((ax1,ax4),(ax2,ax5),(ax3,ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
ax1.plot(x, y, 'k')
ax1.set_title('Periodograms: Primary')
ax1.set_xlim(1/24, max_period)
ax4.set_xlim(1/24, max_period)
ax2.plot(x, y2, 'k')
ax2.set_title('Secondary')
ax3.plot(x, y*y2, 'k')
ax3.set_title('Product Periodogram')
ax4.plot(x, y3, 'k')
ax4.set_title('Primary Data Window')
ax5.plot(x, y4, 'k')
ax5.set_title('Secondary Data Window')
ax6.plot(x, y3*y4, 'k')
ax6.set_title('Product Data Window')
ax3.set_xlabel('Period (days)', size='15')
ax6.set_xlabel('Period (days)', size='15')
ax2.set_ylabel('Normalized Lomb-Scargle Power', size='20')
fig.set_figheight(10)
fig.set_figwidth(15)
plt.savefig(filename + 'periodogram.png')
print('Periodogram peaks above a power of 0.7:', maxima(power_cutoff, x, y, y2))

#plot periodogram - data window
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(x, y*y2, 'b', alpha = 0.5)
ax.plot(x, y3*y4, 'r', alpha = 0.5)
ax.plot(x, y*y2-y3*y4, 'k', alpha = 1)
ax.set_ylim(0,1)
ax.set_title('')
plt.savefig(filename + ' adjusted periodogram.png')

#plot phased RVs
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(phases(maxima(power_cutoff, x, y, y2)[-1], JDp), RVp, 'k.')
ax.plot(phases(maxima(power_cutoff, x, y, y2)[-1], JDs), RVs, 'r.')
ax.set_title('')
ax.set_xlabel('Orbital Phase', size='15')
ax.set_ylabel('Radial Velocity', size='20')
plt.savefig(filename + ' RV-phase diagram.png')

#-----------------------MCMC------------------------#

#constrain parameters
lower_bounds = [0, 0, 0, JD[0]+((JD[-1]-JD[0])/2)-0.75*4.42, 4.43, -75]
upper_bounds = [100, 0.9, 2*np.pi, JD[0]+((JD[-1]-JD[0])/2)+0.75*4.42, 4.45, -55]

#take a walk
sampler = MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 1)

#save the results of the walk
samples = sampler.chain[:, :, :].reshape((-1, 6))
results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                        zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

#create the corner plot
fig = corner.corner(samples, labels=["$K$", "$e$", "$\omega$", "$T$", "$P$", "$\gamma$"],
                    range=[[lower_bounds[0], upper_bounds[0]], [lower_bounds[1],upper_bounds[1]],
                             [lower_bounds[2], upper_bounds[2]],
                             [lower_bounds[3], upper_bounds[3]], [lower_bounds[4], upper_bounds[4]],
                             [lower_bounds[5], upper_bounds[5]]],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18})
fig.savefig(filename + " parameter_results.png")

#create the walkers plot
fig, ax = plt.subplots(6, 1, sharex='col')
for i in range(6):
    for j in range(len(sampler.chain[:, 0, i])):
        ax[i].plot(np.linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.2)
    ax[i].plot(np.linspace(0, nsteps, num=nsteps) , np.ones(nsteps)*results[i][0], 'b', lw=2)
fig.set_figheight(20)
fig.set_figwidth(15)
plt.savefig(filename + ' walk_results.png')

#create the curves plot
x = np.linspace(0, 15.8, num=nsteps)
fig, ax = plt.figure(figsize=(15,8)), plt.subplot(111)
primary, secondary = RV(x, mass_ratio, [results[0][0], results[1][0], results[2][0],
                                        results[3][0], results[4][0], results[5][0]])
ax.plot(x/results[4][0], primary, 'b', lw=2)
ax.plot(x/results[4][0], secondary, 'r', lw=2)
ax.plot(x, np.ones(len(x))*results[5][0], 'k' , label='Systemic Velocity')
ax.plot(phases(results[4][0], JDp), RVp, 'bs', label='Primary RV Data') #data phased to result period
ax.plot(phases(results[4][0], JDs), RVs, 'rs', label='Secondary RV data')
ax.set_xlim([0,1])
plt.savefig(filename + ' curve_results.png')

#print('Results:')
#for i in range(len(initial_guess)):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])
#t = time.time()
#print('Completed in ', int((t-t0)/60), ' minutes and ', int(((t-t0)/60-int((t-t0)/60))*60), 'seconds.')


#-------------circular---MCMC---------------#

#take a walk
sampler = MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 4, nwalkers, nsteps, 1)

#save the results of the walk
circular_samples = sampler.chain[:, :, :].reshape((-1, 4))
results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(circular_samples, [16, 50, 84], axis=0)))))

#create the corner plot
fig = corner.corner(circular_samples, labels=["$K$", "$T$", "$P$", "$\gamma$"],
                    range=[[lower_bounds[0], upper_bounds[0]], [lower_bounds[1],upper_bounds[1]],
                             [lower_bounds[2], upper_bounds[2]], [lower_bounds[3], upper_bounds[3]]],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18})
plt.savefig(filename + ' no e parameter_results.png')

#create the walkers plot
fig, ax = plt.subplots(4, 1, sharex='col')
for i in range(4):
    for j in range(len(sampler.chain[:, 0, i])):
        ax[i].plot(np.linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.2)
    ax[i].plot(np.linspace(0, nsteps, num=nsteps) , np.ones(nsteps)*results[i][0], 'b', lw=2)
fig.set_figheight(20)
fig.set_figwidth(15)
plt.savefig(filename + ' no e walk_results.png')

#create the curves plot
x = np.linspace(0, 15.8, num=nsteps)
fig, ax = plt.figure(figsize=(15,8)), plt.subplot(111)
primary, secondary = RV(x, mass_ratio, [results[0][0], results[1][0], results[2][0], results[3][0]])
ax.plot(x/results[2][0], primary, 'b', lw=2)
ax.plot(x/results[2][0], secondary, 'r', lw=2)
ax.plot(x, np.ones(len(x))*results[3][0], 'k' , label='Systemic Velocity')
ax.plot(phases(results[2][0], JDp), RVp, 'bs', label='Primary RV Data') #data phased to result period
ax.plot(phases(results[2][0], JDs), RVs, 'rs', label='Secondary RV data')
ax.set_xlim([0,1])
plt.savefig(filename + ' no e curve_results.png')

#print('Results:')
#for i in range(len(initial_guess)):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])
t = time.time()
print('Completed in ', int((t-t0)/60), ' minutes and ', int(((t-t0)/60-int((t-t0)/60))*60), 'seconds.')