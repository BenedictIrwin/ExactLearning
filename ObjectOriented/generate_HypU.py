import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import scipy.integrate as integrate

def real_integrand(x,s): return np.real(x**(s-1)*np.exp(-x**2-x))
def imag_integrand(x,s): return np.imag(x**(s-1)*np.exp(-x**2-x))
def special_int(s):  
  r = integrate.quad(real_integrand, 0, np.inf, args=(s))
  i = integrate.quad(imag_integrand, 0, np.inf, args=(s))
  #return (r[0]+ 1j*i[0],r[1:],i[1:])  
  return r[0]+ 1j*i[0], r[1], i[1]

vec_int = np.vectorize(special_int)


## Generate complex moments?
s1 = np.random.uniform(low = 1, high =3, size = 1000)
s2 = np.random.uniform(low = -np.pi, high = np.pi, size = 1000)
s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
q, qre, qie = vec_int(s)

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


np.save("s_values_HypU",s)
np.save("moments_HypU",q)
np.save("logmoments_HypU",np.log(q))
np.save("real_error_HypU",qre)
np.save("imag_error_HypU",qie)
