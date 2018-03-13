#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
#rcParams.update({'figure.autolayout' : True})

# Select the file.
file     = 'data/0611+3325/0611+3325.tbl'

# Create the data variable.
data       = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3, 4, 5))

# Extract the shorthand name.
system         = file.replace('.tbl', '')[5:14]

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[3] for datum in data]
p_err, s_err    = [datum[2] for datum in data], [datum[4] for datum in data]
JDp, JDs        = JD, JD
period_samples  = 10000
max_period      = 2.64
nwalkers, nsteps= 4000, 2000 #minimum nwalker: 14, minimum nsteps determined by the convergence cutoff
cutoff          = 1000

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson  = f.periodogram, f.dataWindow, f.phases, f.wilson
adjustment, RV, residuals, MCMC, walkers = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers
corner, massLimit, coverage, transform   = f.corner, f.massLimit, f.coverage, f.transform

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilson plot (mass ratio)
mass_ratio, intercept, standard_error = wilson(data)

fig = plt.figure(figsize=(5,5))

x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]), -mass_ratio*np.array([np.nanmin(RVs),np.nanmax(RVs)])+intercept

plt.errorbar(RVs, RVp, p_err, s_err, 'k.')
plt.plot(x, y)

#ax.set_title('Wilson plot for 2M17204248+4205070')
plt.text(0, 20, 'q = %s $\pm$ %s' %(round(mass_ratio, 3), round(standard_error, 3)))
plt.ylabel('Primary Velocity ($\\frac{km}{s}$)')#, size='15')
plt.xlabel('Secondary Velocity ($\\frac{km}{s}$)')#, size='15')
plt.title('q = %s $\pm$ %s'%(round(mass_ratio, 3), round(standard_error, 3)))
plt.savefig(file + ' mass ratio.png')
#plt.show()

#check for invalid values
JDp, RVp, p_err = adjustment(JD, RVp, p_err)
JDs, RVs, s_err = adjustment(JD, RVs, s_err)

print(coverage(RVp, RVs))

#calculate periodograms
x, y, delta_x  = periodogram(JDp, RVp, period_samples, max_period)

y2    = periodogram(JDs, RVs, period_samples, max_period)[1]
y3,y4 = dataWindow(JDp, period_samples, max_period)[1], dataWindow(JDs, period_samples, max_period)[1]


#plot periodogram - data window
fig = plt.figure(figsize=(8,3))
#plt.plot(x, y*y2, 'b', alpha = 0.5)
#plt.plot(x, y3*y4, 'r', alpha = 0.5)
plt.plot(x, (y*y2-y3*y4), 'k')
plt.ylabel('Periodogram Power')#, size='15')
plt.xlabel('Period (days)')#, size='15')
plt.ylim(0,1)
plt.xscale('log')
plt.xlim(1,max_period)
plt.title(system)
plt.savefig(file + ' adjusted periodogram.eps')
#plt.show()

plt.close('all')

#-----------------------MCMC------------------------#

import time

start = time.time() #start timer

#constrain parameters
lower_bounds = [0, -0.2, 0, np.median(np.asarray(JD))-0.5*max_period, 2.62, min(min(RVs), min(RVp))]
upper_bounds = [100, 0.2, 2*np.pi, np.median(np.asarray(JD))+0.5*max_period, 2.64, max(max(RVs), max(RVp))]

#take a walk
print('\nwalking...')
sampler = MCMC(mass_ratio, RVp, p_err, RVs, s_err, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 4)
print('Walk complete.\n')

print('Acceptance Fraction: ', np.mean(sampler.acceptance_fraction), '\n')

#save the results of the walk
samples = transform(sampler.chain[:, cutoff:, :].reshape((-1, 6)))

results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                               zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

parms = [x for x in np.transpose(results)[0]]

print('RMS error: ', round(residuals([results[0][0], results[1][0], results[2][0],
                                results[3][0], results[4][0], results[5][0]], mass_ratio, RVp, RVs, JDp, JDs), 3))

print('BIC = %s'%(np.log(len(RVp)+len(RVs))*7 - 2*f.logLikelihood(parms, mass_ratio, RVp, p_err, RVs, s_err, JDp, JDs, lower_bounds, upper_bounds)))

print('Minimum primary mass: ', massLimit(mass_ratio, parms[0], parms[1], parms[-2]), ' Solar masses.\n')

#create walkers plot
print('plotting walk...')
walkers(nsteps, 6, cutoff, sampler).savefig(file + ' %s dimension walk plot.png'%(6), bbox_inches='tight', dpi=300)
plt.close()
print('Walk Plotted\n')

del sampler

#create the corner plot
print('cornering...')
corner(6, samples, parms).savefig(file + ' %s dimension corner plot.eps'%(6), bbox_inches='tight')
plt.close()
print('Corner plotted.\n')

# Write the samples to disk.
print('writing samples to disk...')
np.savetxt(file + ' %s error samples.gz'%(round(residuals(parms, mass_ratio, RVp, RVs, JDp, JDs), 3)),
        samples, delimiter=',')
print('Samples written!\n')

del samples


#write results to console
#print('Results:')
#for i in range(6):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])


#write results to log file
table = open('log.txt', 'a+')
labels = ('K', 'e', 'w', 'T', 'P', 'y')
print('\n' , system, " Results:", file = table)
print('RMS error: ', residuals(np.transpose(results)[0], mass_ratio, RVp, RVs, JDp, JDs), file = table)
print('q  = ', mass_ratio, ' +/-  ', standard_error , file = table)
for i in range(6):
    print(labels[i], ' = ', results[i][0], ' +', results[i][1], ' -', results[i][2], file = table)
table.close()

#end timer
end = time.time()
elapsed = end-start
print('Fitting time was ', int(elapsed), ' seconds.\n')

#-------------circular---MCMC---------------#
start = time.time() #start timer

#take a walk
print('walking...')
sampler = MCMC(mass_ratio, RVp, p_err, RVs, s_err, JDp, JDs, lower_bounds, upper_bounds, 4, nwalkers, nsteps, 4)
print('Walk complete.\n')

print('Acceptance Fraction: ', np.mean(sampler.acceptance_fraction), '\n')

#save the results of the walk
samples = sampler.chain[:, cutoff:, :].reshape((-1, 4))

results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                               zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

parms = [x for x in np.transpose(results)[0]]

print('RMS error: ', round(residuals([parms[0], 0, 0,
                                      parms[1],parms[2], parms[3]], mass_ratio, RVp, RVs, JDp, JDs), 3))

print('BIC = %s'%(np.log(len(RVp)+len(RVs))*7 - 2*f.logLikelihood(parms, mass_ratio, RVp, p_err, RVs, s_err, JDp, JDs, lower_bounds, upper_bounds)))

print('Minimum primary mass: ', massLimit(mass_ratio, parms[0], 0, parms[-2]), ' Solar masses.\n')

#create the walkers plot
print('plotting walk...')
walkers(nsteps, 4, cutoff, sampler).savefig(file + ' %s dimension walk plot.png'%(4), bbox_inches='tight', dpi=300)
plt.close()
print('Walk plotted.\n')

del sampler

# Write the samples to disk.

#create the corner plot
print('cornerning...')
corner(4, samples, parms).savefig(file + ' %s dimension corner plot.eps'%(4), bbox_inches=tight)
plt.close()
print('Corner plotted.\n')

print('writing samples to disk...')
np.savetxt(file + ' %s error samples.gz'%(round(residuals([parms[0], 0, 0, parms[1],parms[2],
                                                           parms[3]], mass_ratio, RVp, RVs, JDp, JDs), 3)),
        samples, delimiter=',')
print('Samples written!\n')

del samples

#write results to console
#print('Results:')
#for i in range(4):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])


#write results to log file
table = open('log.txt', 'a+')
labels = ('K', 'T', 'P', 'y')
print('\n' , system, " Results:", file = table)
print('RMS error: ', residuals([parms[0],0, 0,
                                parms[1],parms[2], parms[3]], mass_ratio, RVp, RVs, JDp, JDs), file = table)
print('q  = ', mass_ratio, ' +/-  ', standard_error , file = table)
for i in range(4):
    print(labels[i], ' = ', results[i][0], ' +', results[i][1], ' -', results[i][2], file = table)
table.close()

#end timer
end = time.time()
elapsed = end-start
print('Fitting time was ', int(elapsed), ' seconds.\n')