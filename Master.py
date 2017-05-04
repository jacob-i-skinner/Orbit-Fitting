#import-libraries-and-data---------------------------------------------------------------------------------------#
import time, sys, corner, os, numpy as np, functions as f
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout' : True})
file     = 'Systems/3325.txt'
data       = np.genfromtxt(file, skip_header=1, usecols=(0, 1, 3))
system         = list(file)

# the string manipulations below extract the 2MASS ID from the file name
while system[0] != '2' and system[1] != 'M':
    del system[0]
while system[-1] != '.':
    del system[-1]
del system[-1]
system = ''.join(system)

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[2] for datum in data]
JDp, JDs        = JD, JD
samples         = 1000
max_period      = 5
power_cutoff    = 0.8
nwalkers, nsteps= 100, 480000
threads         = 4

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, maxima, phases, massRatio = f.periodogram, f.dataWindow, f.maxima, f.phases, f.massRatio
adjustment, RV, residuals, MCMC, lowEFit = f.adjustment, f.RV, f.residuals, f.MCMC, f.lowEFit

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilson plot (mass ratio)
mass_ratio, intercept, r_squared, standard_error, slope_error = massRatio(RVs,RVp, data)
gamma = intercept/(1+mass_ratio)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(RVs, RVp, 'k.')
x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]),-mass_ratio*np.array([np.nanmin(RVs), 
                                                                        np.nanmax(RVs)])+intercept
ax.plot(x, y)
ax.set_title(system)
ax.text(0, 20, 'q = %s $\pm$ %s\n$\gamma$ = %s $\\frac{km}{s}$' %(np.round(mass_ratio, decimals = 3), np.round(standard_error, decimals = 3),
                                                     np.round(gamma, decimals = 3)))
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

#plot periodograms
#fig, ((ax1,ax4),(ax2,ax5),(ax3,ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
#ax1.plot(x, y, 'k')
#ax1.set_title('Periodograms: Primary')
#ax1.set_xlim(1/24, max_period)
#ax4.set_xlim(1/24, max_period)
#ax2.plot(x, y2, 'k')
#ax2.set_title('Secondary')
#ax3.plot(x, y*y2, 'k')
#ax3.set_title('Product Periodogram')
#ax4.plot(x, y3, 'k')
#ax4.set_title('Primary Data Window')
#ax5.plot(x, y4, 'k')
#ax5.set_title('Secondary Data Window')
#ax6.plot(x, y3*y4, 'k')
#ax6.set_title('Product Data Window')
#ax3.set_xlabel('Period (days)', size='15')
#ax6.set_xlabel('Period (days)', size='15')
#ax2.set_ylabel('Normalized Lomb-Scargle Power', size='20')
#fig.set_figheight(10)
#fig.set_figwidth(15)
#plt.savefig(file + 'periodogram.png')


#plot periodogram - data window
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(x, y*y2-y3*y4, 'k', alpha = 1)
ax.plot(x, y*y2, 'b', alpha = 0.5)
ax.plot(x, y3*y4, 'r', alpha = 0.5)
ax.set_ylabel('Periodogram Power')#, size='15')
ax.set_xlabel('Period (days)')#, size='15')
ax.set_ylim(0,1)
ax.set_xlim(delta_x,max_period)
ax.set_title(system)
plt.savefig(file + ' adjusted periodogram.png')
plt.show()
'''
#plot phased RVs
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(phases(maxima(power_cutoff, x, y*y2)[2], JDp), RVp, 'k.')
ax.plot(phases(maxima(power_cutoff, x, y*y2)[2], JDs), RVs, 'r.')
ax.plot(phases(maxima(power_cutoff, x, y*y2)[2], JDp), systemic_velocity*np.ones(len(JDp)))
ax.set_title('Period: %s days' %(maxima(power_cutoff, x, y*y2)[2]))
ax.set_xlabel('Orbital Phase', size='15')
ax.set_ylabel('Radial Velocity', size='20')
plt.show()
#plt.savefig(file + ' RV-phase diagram.png')
'''
#-----------------------MCMC------------------------#
start = time.time() #start timer

#constrain parameters
lower_bounds = [0, -1, 0, JD[0]+((JD[-1]-JD[0])/2)-0.75*3.29, delta_x, min(min(RVs), min(RVp))]
upper_bounds = [200, 1, 2*np.pi, JD[0]+((JD[-1]-JD[0])/2)+0.75*3.29, max_period, max(max(RVs), max(RVp))]

#take a walk
sampler = MCMC(mass_ratio, gamma, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, threads)

#save the results of the walk
samples = sampler.chain[:, 2000:, :].reshape((-1, 6))
results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
parameters = [0,0,0,0,0,0]
for i in range(6):
    parameters[i] = results[i][0]

#Adjust T
T_sampler = lowEFit(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, nwalkers, nsteps, threads, parameters)

#save the results of the adjustment
T_samples = T_sampler.chain[:, 2000:, :].reshape((-1, 1))
T_results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(T_samples, [16, 50, 84], axis=0)))))
results[3], parameters[3] = T_results, T_results[0]

#write results to console
print('Results:')
for i in range(6):
    print(results[i][0], '+',results[i][1], '-',results[i][2])
print('RMS error: ', residuals([results[0][0], results[1][0], results[2][0],
                                results[3][0], results[4][0], results[5][0]], mass_ratio, RVp, RVs, JDp, JDs))

#write results to log file
table = open('log.txt', 'a+')
labels = ('K', 'e', 'w', 'T', 'P', 'y')
print('\n' , system, " Results:", file = table)
print('RMS error: ', residuals([results[0][0], results[1][0], results[2][0],
                                results[3][0], results[4][0], results[5][0]], mass_ratio, RVp, RVs, JDp, JDs), file = table)
print('q  = ', mass_ratio, ' +/-  ', standard_error , file = table)
for i in range(6):
    print(labels[i], ' = ', results[i][0], ' +', results[i][1], ' -', results[i][2], file = table)
table.close()

#end timer
end = time.time()
elapsed = end-start
print('Fitting time was ', int(elapsed), ' seconds.')

#create the corner plot
fig = corner.corner(samples, labels=["$K$", "$e$", "$\omega$", "$T$", "$P$", "$\gamma$"],
                    range=[[lower_bounds[0], upper_bounds[0]], [lower_bounds[1],upper_bounds[1]],
                             [lower_bounds[2], upper_bounds[2]],
                             [lower_bounds[3], upper_bounds[3]], [lower_bounds[4], upper_bounds[4]],
                             [lower_bounds[5], upper_bounds[5]]],
                    #truths = [parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18})
#plt.savefig(file + ' parameter_results.png')


#create the walkers plot
fig, ax = plt.subplots(6, 1, sharex='col')
for i in range(6):
    for j in range(len(sampler.chain[:, 0, i])):
        ax[i].plot(np.linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.2)
    ax[i].plot(np.linspace(0, nsteps, num=nsteps) , np.ones(nsteps)*results[i][0], 'b', lw=2)
fig.set_figheight(20)
fig.set_figwidth(15)
#plt.show()
#plt.savefig(file + ' walk_results.png')


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
plt.title(system)
#plt.savefig(file + ' curve_results.png')
#plt.show()


#-------------circular---MCMC---------------#

#take a walk
sampler = MCMC(mass_ratio, gamma, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 4, nwalkers, nsteps, 4)

#save the results of the walk
circular_samples = sampler.chain[:, :, :].reshape((-1, 4))
results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(circular_samples, [16, 50, 84], axis=0)))))

print('Results:')
for i in range(4):
    print(results[i][0], '+',results[i][1], '-',results[i][2])
print('RMS error: ', residuals([results[0][0], results[1][0],
                                results[2][0], results[3][0]], mass_ratio, RVp, RVs, JDp, JDs))

#write results to log file
table = open('log.txt', 'a+')
labels = ('K', 'T', 'P', 'y')
print('\n' , system, " Results:", file = table)
print('RMS error: ', residuals([results[0][0], results[1][0], results[2][0],
                                results[3][0]], mass_ratio, RVp, RVs, JDp, JDs), file = table)
print('q  = ', mass_ratio, ' +/-  ', standard_error , file = table)
for i in range(4):
    print(labels[i], ' = ', results[i][0], ' +', results[i][1], ' -', results[i][2], file = table)
table.close()


#create the corner plot
fig = corner.corner(circular_samples,labels=['$K$','$T$','$P$','$\gamma$'],
                        range=  [[lower_bounds[0],upper_bounds[0]],
                                 [lower_bounds[1],upper_bounds[1]],
                                 [lower_bounds[2],upper_bounds[2]],
                                 [lower_bounds[3],upper_bounds[3]]],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18})
plt.savefig(file + ' no e parameter_results.png')

#create the walkers plot
fig, ax = plt.subplots(4, 1, sharex='col')
for i in range(4):
    for j in range(len(sampler.chain[:, 0, i])):
        ax[i].plot(np.linspace(0, nsteps, num=nsteps), sampler.chain[j, :, i], 'k', alpha=0.2)
    ax[i].plot(np.linspace(0, nsteps, num=nsteps) , np.ones(nsteps)*results[i][0], 'b', lw=2)
fig.set_figheight(20)
fig.set_figwidth(15)
#plt.savefig(file + ' no e walk_results.png')

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
plt.title(residuals([results[0][0], results[1][0],
                     results[2][0], results[3][0]], mass_ratio, RVp, RVs, JDp, JDs))
plt.savefig(file + ' no e curve_results.png')
plt.show()