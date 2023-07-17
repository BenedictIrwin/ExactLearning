import h5py
from matplotlib import pyplot as plt
h5 = h5py.File('TestData.h5','r')

#<HDF5 dataset "Dataset1": shape (100, 60000, 2), type "<f8">

# 3 looks interesting
# 5


for i in range(5,6):
    #TODO: Fit an inverse Chi distribution
    # Piecewise[{{(x^(-1))^(1 + k/2)/(2^(k/2)*E^(1/(2*x))*Gamma[k/2]), x > 0}}, 0]

    x = h5['Dataset1'][i][:,0] 
    y = h5['Dataset1'][i][:,1]
    plt.plot(x,y,label = f'{i}')
    plt.plot([min(x),max(x)],[0,0],'k:')
    plt.legend()
    plt.show()