import numpy as np
from matplotlib import pyplot as plt

const_dict = {0:0.03462660117004024,
1:-0.5834159435520458,
2:-0.4342316348360242,
3:0.3145818550158719,
4:-0.30679793985255444,
5:1.5333410026073802,
6:-0.997703709023412,
7:-0.6651507936412671,
8:0.5268776790946506,
9:0.22472415832552883,
10:0.6214740741476987,
11:-1.1895055672490606,
12:-0.3386731211432181,
13:-0.9678771952026377,
14:0.8579899176809347,
15:0.8199882381351165,
16:0.018018124124163926,
17:-0.11508605356391755,
18:-0.46357828719711947,
19:0.43215478686631226}


for i in range(2):
    data = np.load(f"fundamental_{i}.npy")
    plt.plot(data[0],data[1],label = f'{i}')
    y =data[1]
    x = data[0]
    a = (y[-2]-y[-1])/(x[-2]-x[-1])
    c = y[-1] - x[-1]*a
    plt.plot(x, a*x + c, 'k:')
    print(a,c)
plt.grid()
plt.legend()
plt.show()

import h5py
h5 = h5py.File('TestData.h5','r')



# from exact_learning.data import points_to_moments
idx = []
for i in range(2):
    x = h5['Dataset1'][i][:,0]
    y = h5['Dataset1'][i][:,1]
    a = []
    b = []
    Y = np.log(y/(y/x)[0])
    for j in range(len(x)-1):
        a.append((Y[j+1]-Y[j])/(x[j+1]-x[j]))
        b.append( Y[j] - a[-1]*x[j])
    print(i,(y/x)[0])
    plt.plot(x,np.log(y/(y/x)[0]),label = f'{i}')
    plt.plot(x[:-1],np.clip(a,-10,10),label = f'a{i}')
    plt.plot(x[:-1],np.clip(b,-10,10),label = f'b{i}')
    if(i == 0):
        plt.plot(x, -2.189 * x + 2.48, 'k:')
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(x, np.gradient(np.log(-y),x))
    plt.plot(x, -3 + 1/(x+0.1)**3)
    #plt.plot(x,np.exp( -2.189 * x + 2.48), 'k:')
    #plt.plot(x,np.log(1.19566 * x**(0.185475) * np.exp(-2.62071 * x)))
    #plt.plot(x,np.log(1.8875 * x**(0.555572) * np.exp(-3.13749 * x)/3.5))
    plt.show()
    




plt.grid()
plt.legend()
plt.show()