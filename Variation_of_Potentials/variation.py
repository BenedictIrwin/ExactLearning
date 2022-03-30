import numpy as np

### Cite the python notebook example

hbar=1
m=1
omega=1
N = 2015
a = 3.0
x = np.linspace(-a/2.,a/2.,N)
h = x[1]-x[0] # Should be equal to 2*np.pi/(N-1)

tag = "abslin_1000"
powe = 1

#tag = "zero"

if(tag=="circ"):
  V = np.array([ -2 * np.sqrt((xx+1)/2)*np.sqrt(1 - (xx+1)/2) if np.abs(xx)<=1 else 1e40 for xx in x])
elif(tag=="zero"):
  V = np.array([ 0 if np.abs(xx)<=1 else 1e40 for xx in x])
elif(tag == "abslin"):
  V = np.array([ np.power(np.abs(xx),powe) - 1 if np.abs(xx)<=1 else 1e40 for xx in x])
  tag = tag + "_" + str(powe)
elif(tag == "abslin_100"):
  V = np.array([ 100*np.power(np.abs(xx),powe) - 100 if np.abs(xx)<=1 else 1e40 for xx in x])
  tag = tag + "_" + str(powe)
elif(tag == "abslin_1000"):
  V = np.array([ 1000*(np.power(np.abs(xx),powe) - 1) if np.abs(xx)<=1 else 1e40 for xx in x])
  tag = tag + "_" + str(powe)

# V[N/2]=2/h   # This would add a "delta" spike in the center.
Mdd = 1./(h*h)*(np.diag(np.ones(N-1),-1) -2* np.diag(np.ones(N),0) + np.diag(np.ones(N-1),1))
H = -(hbar*hbar)/(2.0*m)*Mdd + np.diag(V) 
En,psiT = np.linalg.eigh(H) # This computes the eigen values and eigenvectors
psi = np.transpose(psiT) 
# The psi now contain the wave functions ordered so that psi[n] if the n-th eigen state.

print(En)
exit()

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

order = 0
sign = 1

kk = int((N-1)/2)
#plt.plot(x[kk:],(psi[0]/np.sqrt(h))[kk:],label="$E_{}$={:3.1f}".format(0,En[0]))
psi_inp = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:] , kind='cubic')
psi_inp_lin = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:])

wfn = sign*psi[order]/np.sqrt(h)

plt.plot(x,wfn)
plt.plot(x,np.cos(np.pi*x/2))
#plt.plot(x,np.sqrt(x)*np.sqrt(1-x))
plt.plot(x,V)
plt.xlim((-1.1,1.1))
plt.ylim((-1,2))
plt.show()

plt.plot(np.maximum(np.minimum(V,0),-1),wfn)
plt.show()
plt.plot(np.maximum(np.minimum(V,0),-1),wfn - np.cos(np.pi*x/2))
plt.show()


np.save("pot_{}".format(tag),V)
np.save("wfn_{}".format(tag),wfn)
np.save("spc_{}".format(tag),x)

exit()


x_max = 1

def real_integrand(x,s): return np.real(x**(s-1)*psi_inp(x))
def imag_integrand(x,s): return np.imag(x**(s-1)*psi_inp(x))
def special_int(s):  
  r = integrate.quad(real_integrand, 0, x_max, args=(s))
  i = integrate.quad(imag_integrand, 0, x_max, args=(s))
  return r[0]+ 1j*i[0]
vec_int = np.vectorize(special_int)

def real_integrand_lin(x,s): return np.real(x**(s-1)*psi_inp_lin(x))
def imag_integrand_lin(x,s): return np.imag(x**(s-1)*psi_inp_lin(x))
def special_int_lin(s):  
  r = integrate.quad(real_integrand_lin, 0, x_max, args=(s))
  i = integrate.quad(imag_integrand_lin, 0, x_max, args=(s))
  return r[0]+ 1j*i[0]
vec_int_lin = np.vectorize(special_int_lin)

N_s = 30

## Generate complex moments?
s1 = np.random.uniform(low = 1, high =1, size = N_s)
s2 = np.random.uniform(low = -5*np.pi, high = 5*np.pi, size = N_s)
s = [ 1.0+ t2*1j for t1,t2 in zip(s1,s2) ]
q = vec_int(s)

plt.plot(np.imag(s),np.real(q),'o',label='real')
plt.plot(np.imag(s),np.imag(q),'o',label='imag')
plt.legend()
plt.show()

from scipy.optimize import fsolve


am = np.argmin(np.real(q))
print(am)
print(q[am])

##Look for a zero along the complex axis
def real_int(s): return np.real(special_int(1+s*1j))

res = fsolve(real_int,x0=np.imag(q[am]))
print(res)

exit()


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


np.save("s_values_Circ",s)
np.save("moments_Circ",q)
np.save("logmoments_Circ",np.log(q))

