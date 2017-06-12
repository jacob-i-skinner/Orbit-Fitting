import numpy as np
from matplotlib import pyplot as plt

samples = np.loadtxt('Systems/3325.txt6 emcee samples.gz', delimiter=',', usecols=(0))
print(samples[0])
plt.hist(samples)
plt.show()