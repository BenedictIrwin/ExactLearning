import numpy as np

### Cite the python notebook example

hbar=1
m=1
omega=1
N = 1015
a = 18.0
x = np.linspace(-a/2.,a/2.,N)
h = x[1]-x[0] # Should be equal to 2*np.pi/(N-1)
V = .5*m*omega*x*x 
# V[N/2]=2/h   # This would add a "delta" spike in the center.
Mdd = 1./(h*h)*(np.diag(np.ones(N-1),-1) -2* np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1))
H = -(hbar*hbar)/(2.0*m)*Mdd + np.diag(V) 
En,psiT = np.linalg.eigh(H) # This computes the eigen values and eigenvectors
psi = np.transpose(psiT) 
# The psi now contain the wave functions ordered so that psi[n] if the n-th eigen state.

notok=False
for n in range(len(psi)):
  # s = np.sum(psi[n]*psi[n])
  s = np.linalg.norm(psi[n])  # This does the same as the line above.
  if np.abs(s - 1) > 0.00001: # Check if it is different from one.
    print("Check")
    notok=True

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import k0, gamma
from scipy.interpolate import interp1d
import scipy.integrate as integrate

order = 4
sign = -1

kk = int((N-1)/2)
#plt.plot(x[kk:],(psi[0]/np.sqrt(h))[kk:],label="$E_{}$={:3.1f}".format(0,En[0]))
psi_inp = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:] , kind='cubic')
psi_inp_lin = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:])

plt.plot(x,10*sign*psi[order]/np.sqrt(h)+En[4],label = "$\psi_4$(x)")
plt.plot(x,0.1*V,label="V(x)")
plt.xlabel("x")
plt.ylabel("V(x) or $\psi$(x)")
plt.legend()
plt.show()

x_max = 9

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

N_s = 5000

## Generate complex moments?
s1 = np.random.uniform(low = 1, high =6, size = N_s)
s2 = np.random.uniform(low = -0.9*np.pi, high = 0.9*np.pi, size = N_s)
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


np.save("s_values_Harmonic_4",s)
np.save("moments_Harmonic_4",q)
np.save("logmoments_Harmonic_4",np.log(q))
np.save("real_error_Harmonic_4",qre)
np.save("imag_error_Harmonic_4",qie)
