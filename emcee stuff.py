#this is pretty messy, I intend to go through and cleanup / annotate soon
#This code is built from the example found here: http://dan.iel.fm/emcee/current/user/line/#results
import numpy as np
from matplotlib import pyplot as plt
import functions as f
filename     = '2M17204248+4205070.tbl'
system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))
JD, RVp, RVs = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
JDp, JDs     = JD, JD

periodogram = f.periodogram
dataWindow  = f.dataWindow
maxima      = f.maxima
phases      = f.phases
massRatio   = f.massRatio
adjustment  = f.adjustment
#RV          = f.RV
residuals   = f.residuals

for i in range(0, len(JD)-1):
    if np.isnan(system[i][1]):
        JDp, RVp = adjustment(JD, RVp)
        break
        
for i in range(0, len(JD)-1):
    if np.isnan(system[i][2]):
        JDs, RVs = adjustment(JD, RVs)
        break

mass_ratio = 0.898896068135
Period = 15.8045

def RV(x, mass_ratio, parameters): #function generates RV values plot from given parameters
    check = 1    
    K, e, w, T, P, y = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
    M = (2*np.pi/P)*(x-T) #Mean Anomaly is a function of time
    E1 = M + e*np.sin(M) + ((e**2)*np.sin(2*M)/2) #Eccentric Anomaly is a function of Mean Anomaly
    while True: #iteratively refines estimate of E1 from initial estimate
        E0    = E1
        M0    = E0 - e*np.sin(E0)
        E1    = E0 +(M-M0)/(1-e*np.cos(E0))
        if np.amax(E1-E0) < 1E-9 or check-np.amax(E1-E0) == 0:
            break
        else:
            check = np.amax(E1-E0)
    nu = 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E1/2)) #True Anomaly is a function of Eccentric anomaly
    p, s = (K*(np.cos(nu+w) + (e*np.cos(w)))+y), ((-K/mass_ratio)*(np.cos(nu+w) + (e*np.cos(w)))+y)
    return p, s


x = np.linspace(0, 3.26, num=1000)
plt.figure(figsize=(15,8))
#for K, e, w, T, P, y in samples[np.random.randint(len(samples), size=250)]:
#    parameters = K, e, w, T, P, y
#    primary, secondary = RV(x, mass_ratio, parameters)
#    plt.plot(x/parameters[4],   primary, 'c',  label='Potential Primary Curves', alpha=0.2)
#    plt.plot(x/parameters[4], secondary, 'm',  label='Potential Secondary Curves', alpha=0.2)

#plt.plot(x/parameters[4], RV(x, mass_ratio, initial_guess)[0], color="b", lw=2) #initial guess curve

primary, secondary = RV(x, mass_ratio, [57.98, 0.36, 3.36, 2456180.6, 3.26, -6.95])
plt.plot(x/3.26,   primary,         'b', lw=2)
plt.plot(x/3.26, secondary,         'r', lw=2)
plt.plot(x, np.ones(len(x))*-6.95,  'k' ,      label='Systemic Velocity')
plt.plot(phases(3.26, 0, JDp), RVp, 'bs',      label='Primary RV Data') #data phased to result period
plt.plot(phases(3.26, 0, JDs), RVs, 'rs',      label='Secondary RV data')
plt.xlim(0,1)
#plt.ylim(-24.28,72.84)

#plt.savefig('results.pdf')