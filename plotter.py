#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
plt.tick_params(direction='in')
#rcParams.update({'figure.autolayout' : True})

# Select the file.
file     = 'data/2144+4211/2144+4211.tbl'

# Create the data variable.
data       = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3, 4, 5))

# Extract the shorthand name.
system         = file.replace('.tbl', '')[5:14]

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[3] for datum in data]
p_err, s_err    = [datum[2] for datum in data], [datum[4] for datum in data]
JDp, RVp, p_err = adjustment(JD, RVp, p_err)
JDs, RVs, s_err = adjustment(JD, RVs, s_err)

#define-functions------------------------------------------------------------------------------------------------#

periodogram, dataWindow, phases, wilson  = f.periodogram, f.dataWindow, f.phases, f.wilson
adjustment, RV, residuals, MCMC, walkers = f.adjustment, f.RV, f.residuals, f.MCMC, f.walkers
corner, massLimit, coverage, transform   = f.corner, f.massLimit, f.coverage, f.transform

#now-do-things!--------------------------------------------------------------------------------------------------#

#constrain parameters
lower_bounds = [0, -0.1, 0, np.median(np.asarray(JD))-0.5*max_period, 3.28, min(min(RVs), min(RVp))]
upper_bounds = [100, 0.1, 2*np.pi, np.median(np.asarray(JD))+0.5*max_period, 3.32, max(max(RVs), max(RVp))]

mass_ratio = 
parms      = [ , , ,
               , , ]




#create the corner plot
print('cornering...')
corner(6, samples, parms).savefig(file + ' %s dimension corner plot.pdf'%(6), bbox_inches='tight')
plt.close()
print('Corner plotted.\n')

del samples




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
ax1.plot(x/parms[-2], secondary, 'r--', lw=1, label='Secondary Curve')

# Phase data to result period and plot
ax1.errorbar(phases(parms[-2], JDp), RVp, p_err, np.zeros(len(JDp)), 'ko', label='Primary RV Data')
ax1.errorbar(phases(parms[-2], JDs), RVs, s_err, np.zeros(len(JDs)), 'kv', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot((0, 1), np.zeros(2), 'k', lw = 1)
ax2.errorbar(phases(parms[-2], JDp), RVp-RV(JDp, mass_ratio, parms)[0], p_err, np.zeros(len(JDp)), 'bo')
ax2.errorbar(phases(parms[-2], JDs), RVs-RV(JDs, mass_ratio, parms)[1], s_err, np.zeros(len(JDs)), 'rv')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 20)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 20)
ax2.set_ylabel('O - C', fontsize = 20)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
plt.savefig(file + ' curve results.pdf', bbox_inches='tight')

plt.show()