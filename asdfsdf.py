#import-libraries-and-data---------------------------------------------------------------------------------------#
import time
t0 = time.time()
import corner
import numpy as np
import functions as f
from scipy import stats
from matplotlib import pyplot as plt
filename     = 'Systems/2M17204248+4205070/2M17204248+4205070.tbl'
system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
JDp, JDs        = JD, JD
samples         = 1000
max_period      = 20

#define-functions------------------------------------------------------------------------------------------------#

periodogram     = f.periodogram
dataWindow      = f.dataWindow
maxima          = f.maxima
phases          = f.phases
massRatio       = f.massRatio
adjustment      = f.adjustment
RV              = f.RV
residuals       = f.residuals
constraints     = f.constraints
MCMC            = f.MCMC

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

k = 0

while not k == 15 :
    #calculate periodograms
    x, y  = periodogram(JDp, RVp, samples, max_period)
    y2    = periodogram(JDs, RVs, samples, max_period)[1]
    y3,y4 = dataWindow(JDp, samples, max_period)[1], dataWindow(JDs, samples, max_period)[1]

    #plot periodogram - data window
    fig = plt.figure(figsize=(8,3))
    ax = plt.subplot(111)
    #ax.plot(x, y*y2, 'b', alpha = 0.5)
    #ax.plot(x, y3*y4, 'r', alpha = 0.5)
    ax.plot(x, y*y2-y3*y4, 'k', alpha = 1)
    ax.set_ylim(0,1)
    ax.set_xlim(0,max_period)
    ax.set_title('%s data points'%(15-k))
    plt.savefig('/Users/skinnej3/Desktop/%s adjusted periodogram.png'%(15-k))
    RVp = np.delete(RVp, RVp[-1])
    RVs = np.delete(RVs, RVs[-1])
    JDp = np.delete(JDp, JDp[-1])
    JDs = np.delete(JDs, JDs[-1])
    k += 1

t = time.time()
print('Completed in ', int((t-t0)/60), ' minutes and ', int(((t-t0)/60-int((t-t0)/60))*60), 'seconds.')