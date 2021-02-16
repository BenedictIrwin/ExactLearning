import numpy as np
from scipy.special import ellipk
from matplotlib import pyplot as plt

x = np.random.exponential(size=(1000000))
y = np.random.exponential(size=(1000000))

mean = 0.25*np.pi*(x+y)/ellipk((x-y)/(x+y))

plt.hist(mean,bins=500,density=True)

x = np.linspace(0,12,100)
y = 4*x*np.exp(-2*x)
plt.plot(x,y)
plt.show()

## Sample
real_part = np.random.uniform(low=1,high=4,size=100)
imag_part = np.random.uniform(low=-np.pi,high=np.pi,size=100)
s = np.array([ r + 1j*i for r,i in zip(real_part,imag_part)])
moment = np.mean([np.power(mean,ss) for ss in s],axis=1)
tag = "AGM"
np.save("s_values_{}".format(tag),s)
np.save("moments_{}".format(tag),moment)
np.save("real_error_{}".format(tag),np.array([1e-15 for ss in s]))
np.save("imag_error_{}".format(tag),np.array([1e-15 for ss in s]))
