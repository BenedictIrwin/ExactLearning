
import numpy as np
from matplotlib import pyplot as plt

def pdf(x): return (np.sqrt(x+1)-1)/(x+1)**2
def cdf_inv_plus(x): return (3*x-x**2)/(x-1)**2 + 2*np.sqrt(x)/(np.sqrt(x)-1)**2/(np.sqrt(x)+1)**2
def cdf_inv_minus(x): return (3*x-x**2)/(x-1)**2 - 2*np.sqrt(x)/(np.sqrt(x)-1)**2/(np.sqrt(x)+1)**2

size = 100
uniform_sample = np.random.uniform(0,1,size=size)

sample = cdf_inv_plus(uniform_sample)

plt.hist(sample,bins = 100, density = True)
xx = np.linspace(100,40000,300)
ff = pdf(xx)
plt.plot(xx,ff,'k-')
plt.show()

## We know the Mellin Transform comes out as 
##

from scipy.special import gamma
def phi(s): return 2*gamma(3/2-s)*gamma(s)/np.sqrt(np.pi) - gamma(2-s)*gamma(s)

ss = np.linspace(0 + 0.1,1 - 0.1,100)

## Get the expectation for each value of ss
power = np.mean(np.power(np.expand_dims(sample, axis = 1),np.expand_dims(ss, axis = 0)), axis = 0)

print(ss.shape)
print(ss)

print(phi(ss))
print(phi(ss).shape)

print(phi(1))

def maxx_f(s): return np.amax(sample)**s/size

plt.plot(ss,phi(ss),'r-')
plt.plot(ss,power,'k:')
plt.plot(ss,maxx_f(ss),'b:')
plt.show()
