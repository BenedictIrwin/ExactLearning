import numpy as np
from matplotlib import pyplot as plt
from scipy.special import polygamma

data = np.load('curve_3_2ndorder.npy')
s = np.linspace(-10,10,2000)

def c(x): return np.clip(x,-10,10)

plt.plot(s,data, label='data')
plt.plot(s,c(polygamma(1,s+1.1)), label='polygamma +')
plt.plot(s,c(polygamma(1,-0.7-s)-0.15), label='polygamma -')
plt.legend()
plt.show()