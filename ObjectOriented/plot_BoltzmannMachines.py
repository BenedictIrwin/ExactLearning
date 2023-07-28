from logging import root
import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt
from ObjectOriented import *
import os
from moments import MomentsBundle
from estimator import ExactEstimator

import h5py
from matplotlib import pyplot as plt
h5 = h5py.File('TestData.h5','r')

# from exact_learning.data import points_to_moments
x1 = h5['Dataset1'][3][:,0]
y1 = h5['Dataset1'][3][:,1]
x2 = h5['Dataset1'][1][:,0]
y2 = h5['Dataset1'][1][:,1]

plt.plot(x1,y1,label = 'y1')
plt.plot(x2,y2,label = 'y2')
plt.plot(y1,y2)
plt.legend()
plt.show()

exit()

for i in range(0,100):
    x = h5['Dataset1'][i][:,0] 
    y = h5['Dataset1'][i][:,1]

    if(True):
      plt.plot(x,y,label = f'{i}')
      plt.plot([min(x),max(x)],[0,0],'k:')

#plt.legend()
plt.show()
exit()
"""
if(True):
    try:
      curve_data = np.load(f'curve_{i}_ratio_phi.npy')
    except:
       continue
    plt.plot(curve_data[0],np.clip(curve_data[1],-10,10),label = f'Curve {i}')
"""

plt.legend()
plt.show()
    # Extract moments
    #mb = MomentsBundle(f"curve_{i}")
    #mb.ingest(x, y)
