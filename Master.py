#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
#rcParams.update({'figure.autolayout' : True})
file     = 'Systems/1720+4205/1720+4205.txt'
data       = np.genfromtxt(file, skip_header=1, usecols=(0,1,3))
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
max_period      = 10
nwalkers, nsteps= 1000, 2500 #minimum nwalker: 14, minimum nsteps determined by the convergence cutoff
cutoff          = 500

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson, maximize = f.periodogram, f.dataWindow, f.phases, f.wilson, f.maximize
adjustment, RV, residuals, MCMC, walkers, corner, trim  = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers, f.corner, f.trim

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilson plot (mass ratio)
mass_ratio, intercept, standard_error = wilson(data)
gamma = intercept/(1+mass_ratio)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(RVs, RVp, 'k.')

x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]),-mass_ratio*np.array([np.nanmin(RVs),np.nanmax(RVs)])+intercept

ax.plot(x, y)
#ax.set_title('Wilson plot for 2M17204248+4205070')
ax.text(30, 30, 'q = %s $\pm$ %s\n$\gamma$ = %s $\\frac{km}{s}$' %(round(mass_ratio, 3), round(standard_error, 3),
                                                     round(gamma, 3)))
ax.set_ylabel('Primary Velocity (km/s)')#, size='15')
ax.set_xlabel('Secondary Velocity (km/s)')#, size='15')
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
ax.set_xlim(delta_x,max_period)
ax.set_title(system)
plt.savefig(file + ' adjusted periodogram.png')
#plt.show()

plt.close('all')

#-----------------------MCMC------------------------#

import time

start = time.time() #start timer

#constrain parameters
lower_bounds = [0, -0.9, -1.6, np.median(np.asarray(JD))-0.5*max_period, 3.27, min(min(RVs), min(RVp))]
upper_bounds = [100, 0.9, 2*np.pi, np.median(np.asarray(JD))+0.5*max_period, 3.3, max(max(RVs), max(RVp))]

#take a walk
print('\nwalking...')
sampler = MCMC(mass_ratio, gamma, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 4)
print('Walk complete.\n')

#save the results of the walk
samples = sampler.chain[:, cutoff:, :].reshape((-1, 6))
             

#calculate the parameter values and uncertainties from the quantiles
results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

#create walkers plot
print('plotting walk...')
walkers(nsteps, 6, cutoff, sampler).savefig(file + ' %s dimension walk results.png'%(6))
plt.close()
print('Walk Plotted\n')

del sampler

# delete samples which lie outside of the accepted bounds
print('trimming samples...')
samples = trim(samples, lower_bounds, upper_bounds)
print('Samples trimmed.\n')

# Calculating values.
print('maximizing...')
results = np.transpose(results)
results[0] = maximize(samples)
results = np.transpose(results)
print('Maximization complete.\n')

parms = np.transpose(results)[0]

#create the corner plot
print('cornering...')
corner(6, samples, parms).savefig(file + ' %s dimension parameter results.png'%(6))
plt.close()
print('Corner plotted.\n')

del samples

'''
The T- adjustment step just takes waaaayyy too long. I'm going to try just not using it for a bit. 
If the eccentricity is low enough for it to matter, just use the circular fitter.

#Adjust T
T_sampler = lowEFit(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, nwalkers, nsteps, 4, np.transpose(results)[0])

#save the results of the adjustment
T_samples = T_sampler.chain[:, cutoff:, :].reshape((-1, 1))

T_results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(T_samples, [16, 50, 84], axis=0)))))

results[3][0], results[3][1], results[3][2] = maximize(T_samples)[0], T_results[0][1], T_results[0][2]


samples = np.transpose(samples)
samples[3] = np.transpose(T_samples)
samples = np.transpose(samples)
print('T adjustment complete.')

del T_sampler, T_samples
'''


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
f = plt.figure(figsize=(12,10))
gs = GridSpec(2,1, height_ratios = [3,1])
ax1 = f.add_subplot(gs[0,0])
ax2 = f.add_subplot(gs[1,0])
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)

x = np.linspace(0, parms[4], num=1000)
primary, secondary = RV(x, mass_ratio, parms)

ax1.plot(x/parms[4], primary, 'b', lw=2, label='Primary Curve')
ax1.plot(x/parms[4], secondary, 'r', lw=2, label='Secondary Curve')

ax1.plot(x, np.ones(len(x))*parms[4], 'k' , label='Systemic Velocity')

ax1.plot(phases(parms[4], JDp), RVp, 'ks', label='Primary RV Data') #data phased to result period
ax1.plot(phases(parms[4], JDs), RVs, 'ks', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot(phases(parms[4], JDp), RVp-RV(JDp, mass_ratio, parms)[0], 'bs')
ax2.plot(phases(parms[4], JDs), RVs-RV(JDs, mass_ratio, parms)[1], 'rs')
ax2.plot((0, 1), np.zeros(2), 'k')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 18)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 18)
ax2.set_ylabel('O - C', fontsize = 18)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
plt.savefig(file + ' curve results.png')



#-------------circular---MCMC---------------#



start = time.time() #start timer

#take a walk
print('walking...')
sampler = MCMC(mass_ratio, gamma, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 4, nwalkers, nsteps, 4)
print('Walk complete.\n')

#save the results of the walk
samples = sampler.chain[:, cutoff:, :].reshape((-1, 4))

results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))))

#create the walkers plot
print('plotting walk...')
walkers(nsteps, 4, cutoff, sampler).savefig(file + ' %s dimension walk results.png'%(4))
plt.close()
print('Walk plotted.\n')

del sampler

# delete samples which lie outside of the accepted bounds
print('trimming samples...')
samples = trim(samples, np.delete(lower_bounds, [1,2]), np.delete(upper_bounds, [1,2]))
print('Samples trimmed.\n')

print('maximizing...')
results = np.transpose(results)
results[0] = maximize(samples)
results = np.transpose(results)
print('Maximization complete.\n')

parms = np.transpose(results)[0]

#create the corner plot
print('cornerning...')
corner(4, samples, parms).savefig(file + ' %s dimension parameter results.png'%(4))
plt.close()
print('Corner plotted.\n')

del samples

#write results to console
#print('Results:')
#for i in range(4):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])

print('RMS error: ', round(residuals([results[0][0], 0, 0, results[1][0],
                                      results[2][0], results[3][0]], mass_ratio, RVp, RVs, JDp, JDs), 3))


#write results to log file
table = open('log.txt', 'a+')
labels = ('K', 'T', 'P', 'y')
print('\n' , system, " Results:", file = table)
print('RMS error: ', residuals([results[0][0], 0, 0, results[1][0], results[2][0],
                                results[3][0]], mass_ratio, RVp, RVs, JDp, JDs), file = table)
print('q  = ', mass_ratio, ' +/-  ', standard_error , file = table)
for i in range(4):
    print(labels[i], ' = ', results[i][0], ' +', results[i][1], ' -', results[i][2], file = table)
table.close()

#end timer
end = time.time()
elapsed = end-start
print('Fitting time was ', int(elapsed), ' seconds.\n')

#create the curves plot
f = plt.figure(figsize=(12,10))
gs = GridSpec(2,1, height_ratios = [3,1])
ax1 = f.add_subplot(gs[0,0])
ax2 = f.add_subplot(gs[1,0])
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)

x = np.linspace(0, parms[2], num=1000)
primary, secondary = RV(x, mass_ratio, np.insert(np.transpose(results)[0], 1, [0,0]))

ax1.plot(x/parms[2], primary, 'b', lw=2, label='Primary Curve')
ax1.plot(x/parms[2], secondary, 'r', lw=2, label='Secondary Curve')

ax1.plot(x, np.ones(len(x))*parms[2], 'k' , label='Systemic Velocity')

ax1.plot(phases(parms[2], JDp), RVp, 'ks', label='Primary RV Data') #data phased to result period
ax1.plot(phases(parms[2], JDs), RVs, 'ks', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot(phases(parms[2], JDp), RVp-RV(JDp, mass_ratio, np.insert(parms,1,[0,0]))[0], 'bs')
ax2.plot(phases(parms[2], JDs), RVs-RV(JDs, mass_ratio, np.insert(parms,1,[0,0]))[1], 'rs')
ax2.plot((0, 1), np.zeros(2), 'k')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 18)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 18)
ax2.set_ylabel('O - C', fontsize = 18)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
plt.savefig(file + ' no e curve results.png')