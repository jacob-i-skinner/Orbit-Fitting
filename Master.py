#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
#rcParams.update({'figure.autolayout' : True})
file     = 'Systems/2123+4419/2123+4419.tbl'
data       = np.genfromtxt(file, skip_header=1, usecols=(1,2,3))
system         = list(file)

# the string manipulations below extract the 2MASS ID from the file name
# while system[0] != '2' and system[1] != 'M':
#    del system[0]
while system[-1] != '.':
    del system[-1]
del system[-1]
system = ''.join(system)

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[2] for datum in data]
JDp, JDs        = JD, JD
samples         = 10000
max_period      = 15
nwalkers, nsteps= 1000, 2000 #minimum nwalker: 14, minimum nsteps determined by the convergence cutoff
cutoff          = 1000

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson, maximize = f.periodogram, f.dataWindow, f.phases, f.wilson, f.maximize
adjustment, RV, residuals, MCMC, walkers, corner  = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers, f.corner
uncertainties, massLimit                          = f.uncertainties, f.massLimit

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilson plot (mass ratio)
mass_ratio, intercept, standard_error = wilson(data)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)

x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]), -mass_ratio*np.array([np.nanmin(RVs),np.nanmax(RVs)])+intercept

ax.plot(x, y)
ax.plot(RVs, RVp, 'k.')

#ax.set_title('Wilson plot for 2M17204248+4205070')
ax.text(30, 30, 'q = %s $\pm$ %s' %(round(mass_ratio, 3), round(standard_error, 3)))
ax.set_ylabel('Primary Velocity (km/s)')#, size='15')
ax.set_xlabel('Secondary Velocity (km/s)')#, size='15')
ax.set_title('q = %s $\pm$ %s'%(mass_ratio, standard_error))
plt.savefig(file + ' mass ratio.png')
#plt.show()

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#calculate periodograms
x, y, delta_x  = periodogram(JDp, RVp, samples, max_period)

y2    = periodogram(JDs, RVs, samples, max_period)[1]
y3,y4 = dataWindow(JDp, samples, max_period)[1], dataWindow(JDs, samples, max_period)[1]

#plot periodogram - data window
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(x, y*y2, 'b', alpha = 0.5)
ax.plot(x, y3*y4, 'r', alpha = 0.5)
ax.plot(x, y*y2-y3*y4, 'k', alpha = 1)
ax.set_ylabel('Periodogram Power')#, size='15')
ax.set_xlabel('Period (days)')#, size='15')
ax.set_ylim(0,1)
#ax.set_xscale('log')
ax.set_xlim(delta_x,max_period)
ax.set_title(system)
plt.savefig(file + ' adjusted periodogram.png')
#plt.show()

plt.close('all')

#-----------------------MCMC------------------------#

import time

start = time.time() #start timer

#constrain parameters
lower_bounds = [0, -0.2, 0, np.median(np.asarray(JD))-0.5*max_period, 8.16, min(min(RVs), min(RVp))]
upper_bounds = [100, 0.9, 2*np.pi, np.median(np.asarray(JD))+0.5*max_period, 8.18, max(max(RVs), max(RVp))]


#np.median(np.asarray(JD))-0.5*max_period

#take a walk
print('\nwalking...')
sampler = MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 4)
print('Walk complete.\n')

print('Acceptance Fraction: ', np.mean(sampler.acceptance_fraction), '\n')

#save the results of the walk
samples = sampler.chain[:, cutoff:, :].reshape((-1, 6))

#create walkers plot
print('plotting walk...')
walkers(nsteps, 6, cutoff, sampler).savefig(file + ' %s dimension walk plot.png'%(6))
plt.close()
print('Walk Plotted\n')

del sampler

results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                               zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

parms = [x for x in np.transpose(results)[0]]

print('Minimum primary mass: ', massLimit(mass_ratio, parms[0], parms[1], parms[-2]), ' Solar masses.\n')

# Calculate values.
#print('maximizing...')
#parms = maximize(samples)
#results = uncertainties(parms, mass_ratio, RVp, RVs, JDp, JDs)
#print('Maximization complete.\n')

# Write the samples to disk.
print('writing samples to disk...')
np.savetxt(file + ' %s error samples.gz'%(round(residuals(parms, mass_ratio, RVp, RVs, JDp, JDs), 3)),
        samples, delimiter=',')
print('Samples written!\n')

#create the corner plot
print('cornering...')
corner(6, samples, parms).savefig(file + ' %s dimension corner plot.eps'%(6))
plt.close()
print('Corner plotted.\n')

del samples

#commented out since it was causing unnecessary issues with the interpretation of the walk. It is still valid
#if the eccentricity is negative, perform a transformation of the parameters to make it positive
#add pi to longitude of periastron, and advance time of periastron by period/2
#if results[1][0] < 0:
#    results[1][0], results[2][0], results[3][0] = -results[1][0], results[2][0] + np.pi, results[3][0] + results[4][0]/2
#    results[1][1], results[1][2] = results[1][2], results[1][1] #swap uncertainties of e


#write results to console
#print('Results:')
#for i in range(6):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])

print('RMS error: ', round(residuals([results[0][0], results[1][0], results[2][0],
                                results[3][0], results[4][0], results[5][0]], mass_ratio, RVp, RVs, JDp, JDs), 3))


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
plt.savefig(file + ' curve results.eps')


#-------------circular---MCMC---------------#



start = time.time() #start timer

#take a walk
print('walking...')
sampler = MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 4, nwalkers, nsteps, 4)
print('Walk complete.\n')

print('Acceptance Fraction: ', np.mean(sampler.acceptance_fraction), '\n')

#save the results of the walk
samples = sampler.chain[:, cutoff:, :].reshape((-1, 4))

results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                               zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

parms = [x for x in np.transpose(results)[0]]

print('Minimum primary mass: ', massLimit(mass_ratio, parms[0], 0, parms[-2]), ' Solar masses.\n')

#create the walkers plot
print('plotting walk...')
walkers(nsteps, 4, cutoff, sampler).savefig(file + ' %s dimension walk plot.png'%(4))
plt.close()
print('Walk plotted.\n')

del sampler

# Calculate values.
#print('maximizing...')
#parms = maximize(samples)
#results = uncertainties(parms, mass_ratio, RVp, RVs, JDp, JDs)
#print('Maximization complete.\n')

# Write the samples to disk.
print('writing samples to disk...')
np.savetxt(file + ' %s error samples.gz'%(round(residuals([parms[0], 0, 0, parms[1],parms[2],
                                                           parms[3]], mass_ratio, RVp, RVs, JDp, JDs), 3)),
        samples, delimiter=',')
print('Samples written!\n')

#create the corner plot
print('cornerning...')
corner(4, samples, parms).savefig(file + ' %s dimension corner plot.eps'%(4))
plt.close()
print('Corner plotted.\n')

del samples

#write results to console
#print('Results:')
#for i in range(4):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])


print('RMS error: ', round(residuals([parms[0], 0, 0,
                                      parms[1],parms[2], parms[3]], mass_ratio, RVp, RVs, JDp, JDs), 3))


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
primary, secondary = RV(x, mass_ratio, [parms[0], 0, 0, parms[1], parms[2], parms[3]])

ax1.plot(x, np.ones(len(x))*parms[-1], 'k', lw=1 , label='Systemic Velocity')
ax1.plot(x/parms[-2], primary, 'b', lw=1, label='Primary Curve')
ax1.plot(x/parms[-2], secondary, 'r', lw=1, label='Secondary Curve')

ax1.plot(phases(parms[-2], JDp), RVp, 'k.', label='Primary RV Data') #data phased to result period
ax1.plot(phases(parms[-2], JDs), RVs, 'k.', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot((0, 1), np.zeros(2), 'k', lw = 1)
ax2.plot(phases(parms[-2], JDp), RVp-RV(JDp, mass_ratio, [parms[0], 0, 0, parms[1], parms[2], parms[3]])[0], 'bo')
ax2.plot(phases(parms[-2], JDs), RVs-RV(JDs, mass_ratio, [parms[0], 0, 0, parms[1], parms[2], parms[3]])[1], 'ro')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 20)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 20)
ax2.set_ylabel('O - C', fontsize = 20)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
plt.savefig(file + ' no e curve results.eps')