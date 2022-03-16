import numpy as np
from scipy.special import ai_zeros
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import k0, gamma
from scipy.interpolate import interp1d
import scipy.integrate as integrate

x_max = 15

def real_integrand(x,s): return np.real(x**(s-1)*psi_inp(x))
def imag_integrand(x,s): return np.imag(x**(s-1)*psi_inp(x))
def special_int(s):  
  r = integrate.quad(real_integrand, 0, x_max, args=(s))
  i = integrate.quad(imag_integrand, 0, x_max, args=(s))
  return r[0]+ 1j*i[0], r[1], i[1]
vec_int = np.vectorize(special_int)

#def real_integrand_lin(x,s): return np.real(x**(s-1)*psi_inp_lin(x))
#def imag_integrand_lin(x,s): return np.imag(x**(s-1)*psi_inp_lin(x))
#def special_int_lin(s):  
#  r = integrate.quad(real_integrand_lin, 0, x_max, args=(s))
#  i = integrate.quad(imag_integrand_lin, 0, x_max, args=(s))
#  return r[0]+ 1j*i[0], r[1], i[1]
#vec_int_lin = np.vectorize(special_int_lin)

zeros = np.abs(ai_zeros(20000)[0])
def zeta(s): return np.sum(np.power(zeros,-s))
vec_zeta = np.vectorize(zeta)
N_s = 5000
## Generate complex moments?
s1 = np.random.uniform(low = 3/2+0.01, high =3, size = N_s)
s2 = np.random.uniform(low = -np.pi, high = np.pi, size = N_s)
s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
q = vec_zeta(s)

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


np.save("s_values_AiZeta",s)
np.save("moments_AiZeta",q)
np.save("logmoments_AiZeta",np.log(q))
