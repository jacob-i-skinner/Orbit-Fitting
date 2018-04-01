#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
rcParams.update({'figure.autolayout' : True})

# Select the file.
file     = 'data/0611+3325/0611+3325.tbl'

# Create the data variable.
data       = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3, 4, 5))

# Extract the shorthand name.
system         = file.replace('.tbl', '')[5:14]

'''
This script creates the periodogram/sample histogram comparison plots.
'''

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[3] for datum in data]
p_err, s_err    = [datum[2] for datum in data], [datum[4] for datum in data]
JDp, JDs        = JD, JD
period_samples  = 10000
max_period      = 10
nwalkers, nsteps= 4000, 2000 #minimum nwalker: 14, minimum nsteps determined by the convergence cutoff
cutoff          = 1000

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson  = f.periodogram, f.dataWindow, f.phases, f.wilson
adjustment, RV, residuals, MCMC, walkers = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers
corner, massLimit, coverage, transform   = f.corner, f.massLimit, f.coverage, f.transform

#now-do-things!--------------------------------------------------------------------------------------------------#

#check for invalid values
JDp, RVp, p_err = adjustment(JD, RVp, p_err)
JDs, RVs, s_err = adjustment(JD, RVs, s_err)

# Find mass ratio.
mass_ratio, intercept, standard_error = wilson(data)

#constrain parameters
lower_bounds = [0, -0.1, 0, np.median(np.asarray(JD))-0.5*max_period, 1, min(min(RVs), min(RVp))]
upper_bounds = [100, 0.2, 2*np.pi, np.median(np.asarray(JD))+0.5*max_period, 10, max(max(RVs), max(RVp))]

#take a walk
print('\nwalking...')
sampler = MCMC(mass_ratio, RVp, p_err, RVs, s_err, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 4, period_search=True)
#save just the period results of the walk
samples = transform(sampler.chain[:, cutoff:, :].reshape((-1, 6)))
samples = np.transpose(samples)[4]
print('Walk complete.\n')

#create walkers plot
print('plotting walk...')
walkers(nsteps, 6, cutoff, sampler).savefig(file + ' period walk.png', bbox_inches='tight', dpi=300)
plt.close()
print('Walk Plotted\n')

del sampler

# Calculate periodograms.
x, y, delta_x  = periodogram(JDp, RVp, period_samples, max_period)

y2    = periodogram(JDs, RVs, period_samples, max_period)[1]
y3,y4 = dataWindow(JDp, period_samples, max_period)[1], dataWindow(JDs, period_samples, max_period)[1]


#plot periodogram - data window
#ax.plot(x, y*y2, 'b', alpha = 0.5)
#ax.plot(x, y3*y4, 'r', alpha = 0.5)
plt.figure(figsize=(8,3))
plt.yticks([])
plt.hist(samples, bins='auto', normed=True, color='red')
plt.plot(x, (y*y2-y3*y4), 'k', lw=0.5, alpha=0.75)
plt.xlabel('Period (days)')#, size='15')
plt.ylim(0,1)
plt.xlim(1, max_period)
plt.ylabel('Relative Likelihood')#, size='15')
#plt.savefig(file + ' period plot.pdf', bbox_inches='tight')
plt.show()

plt.close('all')


bound = (2, 3)

plt.hist(samples, bins=400)
plt.xlim(bound)
plt.show()

num = 0
for i in range(4000000):
    if samples[i] > bound[0] and samples[i] < bound[1]:
        num += 1

print(num/4000000)