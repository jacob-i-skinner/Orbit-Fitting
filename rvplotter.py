import numpy as np, functions as f
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
file     = 'Systems/DR13/Tables/2M10464238+1626144.tbl'
data       = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3))
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
nsteps          = 20000
RV              = f.RV
adjustment      = f.adjustment
phases          = f.phases

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#----------------------------------------------------------------------------------------------------#

mass_ratio, parameters = 0.643986286319,[44.5872606179, 0.00355422130935, 0.349973460973,
                                         2456177.21239, 3.28579281142, -7.05424261248]
#----------------------------------------------------------------------------------------------------#

x = np.linspace(0, 15.8, num=nsteps)
fig, ax = plt.figure(figsize=(15,8)), plt.subplot(111)
primary, secondary = RV(x, mass_ratio, parameters)
ax.plot(x/parameters[4], primary, 'b', lw=2)
ax.plot(x/parameters[4], secondary, 'r', lw=2)
ax.plot(x, np.ones(len(x))*parameters[5], 'k' , label='Systemic Velocity')
ax.plot(phases(parameters[4], JDp), RVp, 'bs', label='Primary RV Data') #data phased to result period
ax.plot(phases(parameters[4], JDs), RVs, 'rs', label='Secondary RV data')
ax.set_xlabel('Orbital Phase')
ax.set_ylabel('Radail Velocity $\\frac{km}{s}$')
ax.set_xlim([0,1])
plt.title(system)
plt.savefig(file + ' curve_results.png')
plt.show()