import numpy as np, functions as f, time
from scipy.optimize import minimize
from matplotlib import pyplot as plt

start = time.time()
samples = np.transpose(np.loadtxt('Systems/3325.txt6 emcee samples.gz', delimiter=','))
#samples = np.append(np.random.normal(0, scale=1, size=50000), np.random.normal(3, scale=1, size=25000))
end = time.time()
elapsed = end-start
print('import time was ', int(elapsed), ' seconds.')

'''
x0 = np.arange(20,60,0.01)
x1 = np.arange(-1, 1,0.01)
x2 = np.arange(0,2*np.pi,0.01)
x3 = np.arange(2455927,2455935,0.01)
x4 = np.arange(1,5,0.01)
x5 = np.arange(67,85,0.01)
p0 = f.kernelDensityP(x0, samples[0])
p1 = f.kernelDensityP(x1, samples[1])
p2 = f.kernelDensityP(x2, samples[2])
p3 = f.kernelDensityP(x3, samples[3])
p4 = f.kernelDensityP(x4, samples[4])
p5 = f.kernelDensityP(x5, samples[5])
'''

fig, ax = plt.subplots(6)
for i in range(6):
    ax[i].hist(samples[i], bins='auto', normed=1)
    maximization = minimize(f.kernelDensityP, np.percentile(samples[i], 50), args=samples[i])
    ax[i].plot(maximization.x[0]*np.ones(2), np.linspace(-1, 1, num=2))
    ax[i].plot(np.percentile(samples[0], 50)*np.ones(2), np.linspace(-1, 1, num=2))

#ax[0].plot(x0, p0)
#ax[1].plot(x1, p1)
#ax[2].plot(x2, p2)
#ax[3].plot(x3, p3)
#ax[4].plot(x4, p4)
#ax[5].plot(x5, p5)


#fig.set_figheight(20)
#fig.set_figwidth(10)
plt.show()