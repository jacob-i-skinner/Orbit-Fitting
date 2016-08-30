####### Revisit, check code from current massRatios() ######

import os
import numpy as np
table = open('Mass Ratios.tbl', 'a+')
print('Mass Ratios:', file = table)
for i in range(0,len(os.listdir())-1):
    if '.tbl' in os.listdir()[i]:
        system = np.genfromtxt(os.listdir()[i], skip_header=1, usecols=(0, 1, 2))
        RVp =[datum[1] for datum in system if not np.isnan(datum[1]+datum[2])]
        RVs =[datum[2] for datum in system if not np.isnan(datum[1]+datum[2])]
        m,b = np.polyfit(RVs, RVp, 1)
        #plt.plot(RVs, RVp, 'r.')
        #plt.plot(np.linspace(-80,60, num=10), np.linspace(-80,60, num=10)*m+b)
        print(os.listdir()[i],':', -m, file=table)