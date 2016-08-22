#import-libraries-and-data---------------------------------------------------------------------------------------#
import numpy as np
import functions as f
from matplotlib import pyplot as plt
#filename     = 'Systems/2M03450783+3102335/4586_2M03450783+3102335_rvs.tbl'
#system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))

#define-functions------------------------------------------------------------------------------------------------#

periodogram    = f.periodogram
dataWindow     = f.dataWindow
maxima         = f.maxima
phases         = f.phases
adjustment     = f.adjustment
RV             = f.RV
residuals      = f.residuals
constraints    = f.constraints
constraintsNoE = f.constraintsNoE
alteredRV      = f.alteredRV
initialGuess   = f.initialGuess
noERV          = f.noERV
alteredNoERV   = f.alteredNoERV
initialGuessNoE= f.initialGuessNoE

mr = 0.6 #mass ratio
x = np.array([2540, 2542, 2543, 2546, 2550, 2552, 2552.35, 2555, 2560, 2561, 2562, 2563, 2563.5, 2564, 2564.5, 2564.75, 2565])
parameters = 50, 0.5, 1.3, 2554, 3, 15
RVp, RVs = RV(x, mr, parameters)



#define-variables------------------------------------------------------------------------------------------------#

JD           = x 
JDp, JDs     = JD, JD


#create the curves plot
x = np.linspace(0, 15.8, num=1000)
fig, ax = plt.figure(figsize=(15,8)), plt.subplot(111)

primary, secondary = RV(x, mr, parameters)

ax.plot(x/3, primary, 'm', lw=2)
ax.plot(x/3, secondary, 'm', lw=2)
ax.plot(x, np.ones(len(x))*15, 'k' , label='Systemic Velocity')
ax.plot(phases(3, JDp), RVp, 'ms', label='Primary RV Data') #data phased to result period
ax.plot(phases(3, JDs), RVs, 'ks', label='Secondary RV data')
ax.set_xlim([0,1])
#plt.savefig('curve_results.png')


plt.show()