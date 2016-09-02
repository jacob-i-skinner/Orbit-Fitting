#import-libraries-and-data---------------------------------------------------------------------------------------#
import time
t0 = time.time()
import corner
import numpy as np
import functions as f
from scipy import stats
from matplotlib import pyplot as plt
filename     = 'Systems/DQ Tau/DQ Tau.tbl'
system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))

#define-variables------------------------------------------------------------------------------------------------#

JD, RVp, RVs    = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
JDp, JDs        = JD, JD
samples         = 1000
max_period      = 9
power_cutoff    = 0.24
nwalkers, nsteps= 400, 12000

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

#now-do-things!--------------------------------------------------------------------------------------------------#

#plot Wilkinson plot (mass ratio)
mass_ratio, intercept, r_squared, standard_error, slope_error = massRatio(RVs,RVp, system)

#check for invalid values
JDp, RVp = adjustment(JD, RVp)
JDs, RVs = adjustment(JD, RVs)

#-----------------------MCMC------------------------#

#constrain parameters
lower_bounds = [0, 0, 0, JD[0]+((JD[-1]-JD[0])/2)-0.75*15.8, 15.7, 5]
upper_bounds = [100, 0.9, 2*np.pi, JD[0]+((JD[-1]-JD[0])/2)+0.75*15.8, 15.9, 45]

for k in range(33):

    system       = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2), max_rows=(33-k))
    JD, RVp, RVs    = [datum[0] for datum in system], [datum[1] for datum in system], [datum[2] for datum in system]
    JDp, JDs        = JD, JD
    JDp, RVp = adjustment(JD, RVp)
    JDs, RVs = adjustment(JD, RVs)

    print(system.shape)

    #take a walk
    sampler = MCMC(mass_ratio, RVp, RVs, JDp, JDs, lower_bounds, upper_bounds, 6, nwalkers, nsteps, 4)

    #save the results of the walk
    samples = sampler.chain[:, 2000:, :].reshape((-1, 6))
    results = np.asarray(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
#create the corner plot
    fig = corner.corner(samples,labels=['$K$','$e$','$\omega$','$T$','$P$','$\gamma$'],
                        range=  [[lower_bounds[0],upper_bounds[0]],
                                 [lower_bounds[1],upper_bounds[1]],
                                 [lower_bounds[2],upper_bounds[2]],
                                 [lower_bounds[3],upper_bounds[3]],
                                 [lower_bounds[4],upper_bounds[4]],
                                 [lower_bounds[5],upper_bounds[5]]],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18})
    plt.savefig('Systems/DQ Tau/test/%s corner.png'%(33-k))

    plt.close(fig)

#print('Results:')
#for i in range(len(initial_guess)):
#    print(results[i][0], '+',results[i][1], '-',results[i][2])
#t = time.time()
#print('Completed in ', int((t-t0)/60), ' minutes and ', int(((t-t0)/60-int((t-t0)/60))*60), 'seconds.')

t = time.time()
print('Completed in ', int((t-t0)/60), ' minutes and ', int(((t-t0)/60-int((t-t0)/60))*60), 'seconds.')