import numpy as np, functions as f
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import rcParams

file     = 'data/2123+4419/2123+4419.tbl'
data       = np.genfromtxt(file, skip_header=1, usecols=(1,2,3))
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
RV              = f.RV
adjustment      = f.adjustment
phases          = f.phases

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#----------------------------------------------------------------------------------------------------#

mass_ratio, parms = 0.904,[59.1143202682, 0,  0,
                                    0,    24.695, -123]
#----------------------------------------------------------------------------------------------------#

f = plt.figure(figsize=(11,10))
gs = GridSpec(2,1, height_ratios = [4,1])
ax1 = f.add_subplot(gs[0,0])
ax1.tick_params(labelsize=14)
ax2 = f.add_subplot(gs[1,0])
ax2.tick_params(labelsize=14)
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#f.suptitle('Radial Velocity Curve for ' + system, fontsize = 22)

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


#plt.savefig('2439_RV.eps')
plt.show()