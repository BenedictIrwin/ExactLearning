
import numpy as np
from matplotlib import pyplot as plt

def pdf(x): return (np.sqrt(x+1)-1)/(x+1)**2
def cdf_inv_plus(x): return (3*x-x**2)/(x-1)**2 + 2*np.sqrt(x)/(np.sqrt(x)-1)**2/(np.sqrt(x)+1)**2
def cdf_inv_minus(x): return (3*x-x**2)/(x-1)**2 - 2*np.sqrt(x)/(np.sqrt(x)-1)**2/(np.sqrt(x)+1)**2

size = 100000
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

#ss = np.linspace(0-2 + 0.01,3/2+1 - 0.01,100)
ss = np.linspace(0-1 + 0.01,3/2 - 0.01,100)

## Get the expectation for each value of ss
power = np.mean(np.power(np.expand_dims(sample, axis = 1),np.expand_dims(ss-1, axis = 0)), axis = 0)

print(ss.shape)
print(ss)

print(phi(ss))
print(phi(ss).shape)

print(phi(1))

def maxx_f(s): return np.amax(sample)**(s-1)/size
def minn_f(s): return np.amin(sample)**(s-1)/size
def example_f(s,x): return x**(s-1)/size

plt.plot(ss,phi(ss),'r-',label="Theoretical phi(s)")
plt.plot(ss,power,'k:',label = "Numerical E[x^(s-1)]")
plt.plot(ss,maxx_f(ss),'b:',label = "x_max^(s-1)")
plt.plot(ss,np.exp(np.log(power) - np.amax([np.log(minn_f(ss)),np.log(maxx_f(ss))],axis = 0))/size,'k-',label = "Error Profile")
plt.legend()
plt.show()
plt.plot(ss,np.real(np.log(np.real(phi(ss)))),'r-',label="Theoretical logphi(s)")
plt.plot(ss,np.log(power),'k:',label = "Numerical logE[x^(s-1)]")
plt.plot(ss,np.log(maxx_f(ss)),'b:',label = "log(x_max^(s-1))")
plt.plot(ss,np.log(minn_f(ss)),'b:',label = "log(x_min^(s-1))")
plt.plot(ss,np.log(power) - np.amax([np.log(minn_f(ss)),np.log(maxx_f(ss))],axis = 0),'k-',label = "log(Error Profile)")

#for y in np.linspace(np.amin(sample),np.amin(sample)*3,100):
#  plt.plot(ss,np.log(example_f(ss,y)),'g:')
plt.legend()
plt.show()
plt.plot(ss,phi(ss)-power,'r-',label="Theoretical - Numerical")
plt.plot(ss,0.02*np.exp(np.log(power) - np.amax([np.log(minn_f(ss)),np.log(maxx_f(ss))],axis = 0))/size,'k-',label = "Error Profile")
plt.plot(ss,[0 for q in ss],'k:')
plt.legend()
plt.show()



