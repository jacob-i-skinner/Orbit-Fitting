#import-libraries-and-data---------------------------------------------------------------------------------------#
import os, numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, rcParams
#rcParams.update({'figure.autolayout' : True})

# Select the file.
file    = 'data/2144+4211/2144+4211.tbl'

# Create the data variable.
data    = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3, 4, 5))
source  = np.genfromtxt(file, dtype=str, skip_header=1, usecols=(8))

# Extract the shorthand name.
system  = file.replace('.tbl', '')[5:14]

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[3] for datum in data]
p_err, s_err    = [datum[2] for datum in data], [datum[4] for datum in data]
JDp, RVp, p_err = f.adjustment(JD, RVp, p_err)
JDs, RVs, s_err = f.adjustment(JD, RVs, s_err)

#define-functions------------------------------------------------------------------------------------------------#

RV, phases = f.RV, f.phases

#now-do-things!--------------------------------------------------------------------------------------------------#

# Check for HET values, create data filter to pick them out.
HET = [1 if x == 'HET' else 0 for x in source]

# This is less efficient than it could be, but oh well.
# Separate APOGEE and HET observations.
APO_JDp = np.asarray([JDp[i] for i in range(len(JDp)) if not HET[i]])
APO_JDs = np.asarray([JDs[i] for i in range(len(JDs)) if not HET[i]])
APO_RVp = np.asarray([RVp[i] for i in range(len(RVp)) if not HET[i]])
APO_RVs = np.asarray([RVs[i] for i in range(len(RVs)) if not HET[i]])
APO_p_err = np.asarray([p_err[i] for i in range(len(p_err)) if not HET[i]])
APO_s_err = np.asarray([s_err[i] for i in range(len(s_err)) if not HET[i]])

HET_JDp = np.asarray([JDp[i] for i in range(len(JDp)) if HET[i]])
HET_JDs = np.asarray([JDs[i] for i in range(len(JDs)) if HET[i]])
HET_RVp = np.asarray([RVp[i] for i in range(len(RVp)) if HET[i]])
HET_RVs = np.asarray([RVs[i] for i in range(len(RVs)) if HET[i]])
HET_p_err = np.asarray([p_err[i] for i in range(len(p_err)) if HET[i]])
HET_s_err = np.asarray([s_err[i] for i in range(len(s_err)) if HET[i]])

mass_ratio = 0.946985628187
parms      = [61.1551472716, 0, 0, 2456205.33813, 3.29813071475, -17.2248465232]


#create the curves plot
fig = plt.figure(figsize=(11,10))
gs = GridSpec(2,1, height_ratios = [4,1])
ax1 = fig.add_subplot(gs[0,0])
ax1.tick_params(labelsize=14)
ax2 = fig.add_subplot(gs[1,0])
ax2.tick_params(labelsize=14)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tick_params(direction='in')
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
#fig.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)

x = np.linspace(0, parms[-2], num=1000)
primary, secondary = RV(x, mass_ratio, parms)

ax1.plot(x, np.ones(len(x))*parms[-1], 'k', lw=1 , label='Systemic Velocity')
ax1.plot(x/parms[-2], primary, 'b', lw=1, label='Primary Curve')
ax1.plot(x/parms[-2], secondary, 'r--', lw=1, label='Secondary Curve')

# Phase APOGEE data to result period and plot
ax1.errorbar(phases(parms[-2], APO_JDp), APO_RVp, APO_p_err, np.zeros(len(APO_JDp)), 'ko', label='Primary RV Data')
ax1.errorbar(phases(parms[-2], APO_JDs), APO_RVs, APO_s_err, np.zeros(len(APO_JDs)), 'ks', label='Secondary RV data')

# Plot the APOGEE observed - computed underplot
ax2.plot((0, 1), np.zeros(2), 'k', lw = 1)
ax2.errorbar(phases(parms[-2], APO_JDp), APO_RVp-RV(APO_JDp, mass_ratio, parms)[0], APO_p_err, np.zeros(len(APO_JDp)), 'bo')
ax2.errorbar(phases(parms[-2], APO_JDs), APO_RVs-RV(APO_JDs, mass_ratio, parms)[1], APO_s_err, np.zeros(len(APO_JDs)), 'rs')

# If any HET observations are present, include them.

if not len(HET_JDp+HET_JDs) == 0:
    # Phase HET data to result period and plot
    ax1.errorbar(phases(parms[-2], HET_JDp), HET_RVp, HET_p_err, np.zeros(len(HET_JDp)), 'kv', label='Primary RV Data')
    ax1.errorbar(phases(parms[-2], HET_JDs), HET_RVs, HET_s_err, np.zeros(len(HET_JDs)), 'k^', label='Secondary RV data')

    # Plot the HET observed - computed underplot
    ax2.plot((0, 1), np.zeros(2), 'k', lw = 1)
    ax2.errorbar(phases(parms[-2], HET_JDp), HET_RVp-RV(HET_JDp, mass_ratio, parms)[0], HET_p_err, np.zeros(len(HET_JDp)), 'bv')
    ax2.errorbar(phases(parms[-2], HET_JDs), HET_RVs-RV(HET_JDs, mass_ratio, parms)[1], HET_s_err, np.zeros(len(HET_JDs)), 'r^')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 20)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 20)
ax2.set_ylabel('O - C $\\frac{km}{s}$', fontsize = 20)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
plt.savefig('4211_RV.pdf', bbox_inches='tight')
plt.show()