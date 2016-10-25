#import-libraries-and-data---------------------------------------------------------------------------------------#

import numpy as np
import functions as f
from matplotlib import pyplot as plt

#define-variables------------------------------------------------------------------------------------------------#

#define-functions------------------------------------------------------------------------------------------------#
RV              = f.RV

#--do-stuff--#

x = np.linspace(0,1,num=1000)
mass_ratio = 0.5
K = 1
e = 0
w = 0
T = 0
P = 1
y = 0

parameters = K, e, w, T, P, y

primary, secondary = RV(x, mass_ratio, parameters)
plt.figure(figsize=(10,5))
plt.plot(x, primary)
plt.plot(x, secondary, 'r')
plt.show()