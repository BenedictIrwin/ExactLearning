import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import gamma

## Randomly sample points from k0
## Do this according to the relative "density"
#N = 2000000
#x = sorted(np.random.uniform(low=0,high=1,size=N)) ## draw 100 points between 1 and 2
#
#def func(x): return np.sqrt(np.sqrt(x)*np.sqrt(1-x)) * np.sqrt(1-np.sqrt(x)*np.sqrt(1-x))
#
#func_v = np.vectorize(func)
#
#values = func_v(x)
#sum_values = np.sum(values)
#probs = [ xx/sum_values for xx in values ]
## Randomly select samples from values based on the distribution
#d = np.random.choice(x,N,p=probs,replace=True)

#d= np.abs(np.random.normal(loc=0, scale = 1, size = N))

#if(False):
#  plt.hist(d,bins=100, normed=True)
#  plt.xlabel("Distance")
#  plt.ylabel("Probability Density")
#  plt.show()

#import scipy
#from scipy.integrate import quad
#
#
#
#def complex_quadrature(func, a, b, **kwargs):
#  def real_func(x):
#    return scipy.real(func(x))
#  def imag_func(x):
#    return scipy.imag(func(x))
#  real_integral = quad(real_func, a, b, **kwargs)
#  imag_integral = quad(imag_func, a, b, **kwargs)
#  return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

import scipy.integrate as integrate

def real_integrand(x,s): return np.real(x**(s-1)*np.sqrt(np.sqrt(x)*np.sqrt(1-x))*np.sqrt(1-np.sqrt(x)*np.sqrt(1-x)))
def imag_integrand(x,s): return np.imag(x**(s-1)*np.sqrt(np.sqrt(x)*np.sqrt(1-x))*np.sqrt(1-np.sqrt(x)*np.sqrt(1-x)))

#def real_integrand(x,s): return np.real(x**(s-1)*np.sqrt(x)*np.sqrt(1-x))
#def imag_integrand(x,s): return np.imag(x**(s-1)*np.sqrt(x)*np.sqrt(1-x))
def special_int(s):  
  r = integrate.quad(real_integrand, 0, 1, args=(s))
  i = integrate.quad(imag_integrand, 0, 1, args=(s))
  #return (r[0]+ 1j*i[0],r[1:],i[1:])  
  return r[0]+ 1j*i[0]

vec_int = np.vectorize(special_int)

## Generate complex moments?
s1 = np.random.uniform(low = 0.25, high =3, size = 4000)
s2 = np.random.uniform(low = -2*np.pi, high =2*np.pi, size = 4000)
s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
q = vec_int(s)
a = [ np.sqrt(np.pi)*gamma(ss+1/2)/2/gamma(ss+2) for ss in s ]

print(q)
print(np.array(a))




if(True):
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(np.real(s), np.imag(s), np.real(q), c=np.real(q), cmap='Reds');
  ax.scatter3D(np.real(s), np.imag(s), np.imag(q), c=np.imag(q), cmap='Greens');
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$E[x^{s-1}]$')
  plt.show()
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(np.real(s), np.imag(s), np.real(np.log(q)), c=np.real(np.log(q)), cmap='Reds');
  ax.scatter3D(np.real(s), np.imag(s), np.imag(np.log(q)), c=np.imag(np.log(q)), cmap='Greens');
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$\log E[x^{s-1}]$')
  plt.show()


np.save("s_values_roots",s)
np.save("moments_roots",q)
np.save("logmoments_roots",np.log(q))

