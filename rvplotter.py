import numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
file     = 'Systems/1720+4205/1720+4205.txt'
data       = np.genfromtxt(file, skip_header=1, usecols=(0, 1, 3))
system         = list(file)

# the string manipulations below extract the 2MASS ID from the file name
while system[0] != '2' and system[1] != 'M':
    del system[0]
while system[-1] != '.':
    del system[-1]
del system[-1]
system = ''.join(system)

JD, RVp, RVs    = [datum[0] for datum in data], [datum[1] for datum in data], [datum[2] for datum in data]
JDp, JDs        = JD, JD
power_cutoff    = 0.8
RV              = f.RV
adjustment      = f.adjustment
phases          = f.phases

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#----------------------------------------------------------------------------------------------------#

mass_ratio, parameters = 0.657390338631,[43.3942298488, 0, 0, 56048.8891775, 3.28649612275, -6.71347635903]
#----------------------------------------------------------------------------------------------------#

f = plt.figure(figsize=(12,10))
gs = GridSpec(2,1, height_ratios = [3,1])
ax1 = f.add_subplot(gs[0,0])
ax2 = f.add_subplot(gs[1,0])
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)


x = np.linspace(0, parameters[4], num=1000)
primary, secondary = RV(x, mass_ratio, parameters)

ax1.plot(x/parameters[4], primary, 'b', lw=2, label='Primary Curve')
ax1.plot(x/parameters[4], secondary, 'r', lw=2, label='Secondary Curve')

ax1.plot(x, np.ones(len(x))*parameters[4], 'k' , label='Systemic Velocity')

ax1.plot(phases(parameters[4], JDp), RVp, 'ks', label='Primary RV Data') #data phased to result period
ax1.plot(phases(parameters[4], JDs), RVs, 'ks', label='Secondary RV data')

# Plot the observed - computed underplot
ax2.plot(phases(parameters[4], JDp), RVp-RV(JDp, mass_ratio, parameters)[0], 'bs')
ax2.plot(phases(parameters[4], JDs), RVs-RV(JDs, mass_ratio, parameters)[1], 'rs')
ax2.plot((0, 1), np.zeros(2), 'k')

# Adjust the look of the plot
plt.xlabel('Orbital Phase', fontsize = 18)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 18)
ax2.set_ylabel('O - C', fontsize = 18)
ax1.set_xlim([0,1])
ax2.set_xlim([0,1])


#plt.savefig('4205_RV.pdf')
plt.show()