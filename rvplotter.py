import numpy as np, functions as f
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
file     = 'Systems/4205.txt'
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

mass_ratio, parameters = 0.657390338631,[43.3942298488, 56048.8891775, 3.28649612275, -6.71347635903]
#----------------------------------------------------------------------------------------------------#

x = np.linspace(0, parameters[2], num=1000)
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, figsize=(12,12))

#modified to plot multiple curves
primary, secondary = RV(x, mass_ratio, parameters)

ax1.plot(x/parameters[2], primary, 'b', lw=2)
ax1.plot(x/parameters[2], secondary, 'r', lw=2)

ax1.plot(x, np.ones(len(x))*parameters[3], 'k' , label='Systemic Velocity')

ax1.plot(phases(3.28649612275, JDp), RVp, 'ks', label='Primary RV Data') #data phased to result period
ax1.plot(phases(3.28649612275, JDs), RVs, 'ks', label='Secondary RV data')

ax2.plot(phases(3.28649612275, JDp), RVp-RV(JDp, mass_ratio, parameters)[0], 'bs')
ax2.plot(phases(3.28649612275, JDs), RVs-RV(JDs, mass_ratio, parameters)[1], 'rs')


plt.xlabel('Orbital Phase', fontsize = 18)
ax1.set_ylabel('Radial Velocity $\\frac{km}{s}$', fontsize = 18)
ax2.set_ylabel('O - C', fontsize = 18)
ax1.set_xlim([0,1])
plt.subplots_adjust(wspace=0, hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.title('Radial Velocity Curve for 2M06115599+3325505', fontsize = 18)
plt.savefig('4205_RV.pdf')
plt.show()