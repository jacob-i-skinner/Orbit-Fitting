#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
rcParams.update({'figure.autolayout' : True})

# Select the file.
file     = 'data/2144+4211/2144+4211.tbl'

# Create the data variable.
data       = np.genfromtxt(file, skip_header=1, usecols=(1,2,3))

# Extract the shorthand name.
system         = file.replace('.tbl', '')[5:14]

'''
This script creates the periodogram/sample histogram comparison plots.
'''

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[2] for datum in data]
JDp, JDs        = JD, JD
max_period      = 100
period_samples  = 10000

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson, maximize = f.periodogram, f.dataWindow, f.phases, f.wilson, f.maximize
adjustment, RV, residuals, MCMC, walkers, corner  = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers, f.corner
uncertainties, massLimit, coverage                = f.uncertainties, f.massLimit, f.coverage

#now-do-things!--------------------------------------------------------------------------------------------------#

samples = np.loadtxt('data/2144+4211/2144+4211.tbl 4.607 error samples.txt', delimiter=',', usecols=(2), dtype=float)

#plot Wilson plot (mass ratio)
mass_ratio, intercept, standard_error = wilson(data)

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#calculate periodograms
x, y, delta_x  = periodogram(JDp, RVp, period_samples, max_period)

y2    = periodogram(JDs, RVs, period_samples, max_period)[1]
y3,y4 = dataWindow(JDp, period_samples, max_period)[1], dataWindow(JDs, period_samples, max_period)[1]


#plot periodogram - data window
#ax.plot(x, y*y2, 'b', alpha = 0.5)
#ax.plot(x, y3*y4, 'r', alpha = 0.5)
plt.figure(figsize=(8,3))
plt.yticks([])
plt.hist(samples, bins=200, normed=True, color='red')
plt.plot(x, (y*y2-y3*y4)*15, 'k', lw=0.5, alpha=0.75)
plt.xlabel('Period (days)')#, size='15')
plt.ylim(0,10)
plt.xlim(1, 20)
plt.ylabel('Relative Likelihood')#, size='15')
plt.savefig(file + ' period plot.eps')
plt.show()

plt.close('all')

'''
bound = (3.29, 3.305)

plt.hist(samples, bins=40000)
plt.xlim(bound)
plt.show()

num = 0
for i in range(4000000):
    if samples[i] > bound[0] and samples[i] < bound[1]:
        num += 1

print(num/4000000)
'''