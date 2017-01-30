#import-libraries-and-data---------------------------------------------------------------------------------------#
import numpy as np, functions as f
from matplotlib import pyplot as plt
filename     = 'Systems/DR13/Tables/2M21442066+4211363.tbl'
system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
JDp, JDs        = JD, JD
samples         = 1000
max_period      = 7.5
power_cutoff    = 0.8
nwalkers, nsteps= 400, 10000

#define-functions------------------------------------------------------------------------------------------------#

periodogram     = f.periodogram
dataWindow      = f.dataWindow
maxima          = f.maxima
phases          = f.phases
massRatio       = f.massRatio
adjustment      = f.adjustment


#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilkinson plot (mass ratio)
mass_ratio, intercept, r_squared, standard_error, slope_error = massRatio(RVs,RVp, system)
systemic_velocity = intercept/(1+mass_ratio)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(RVs, RVp, 'k.')
x, y = np.array([np.nanmin(RVs), np.nanmax(RVs)]),-mass_ratio*np.array([np.nanmin(RVs), 
                                                                        np.nanmax(RVs)])+intercept
ax.plot(x, y)
ax.set_ylabel('Primary Velocity', size='15')
ax.set_xlabel('Secondary Velocity', size='15')
plt.savefig(filename + ' mass ratio.png')
#plt.show()

print('mass ratio is ', mass_ratio, "+/-", standard_error, '\nsystemic velocity is ', systemic_velocity)

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
#plt.savefig(filename + 'periodogram.png')


#plot periodogram - data window
fig = plt.figure(figsize=(8,3))
ax = plt.subplot(111)
ax.plot(x, y*y2-y3*y4, 'k', alpha = 1)
ax.plot(x, y*y2, 'b', alpha = 0.5)
ax.plot(x, y3*y4, 'r', alpha = 0.5)
ax.set_ylim(0,1)
ax.set_xlim(delta_x,max_period)
ax.set_title('')
print('Periodogram peaks above a power of %s:' % (power_cutoff), maxima(power_cutoff, x, y*y2-y3*y4))
#plt.show()
plt.savefig(filename + ' adjusted periodogram.png')
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
#plt.savefig(filename + ' RV-phase diagram.png')
'''