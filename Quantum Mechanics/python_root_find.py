import numpy as np
from scipy.special import gamma, lambertw
from scipy.optimize import minimize

### Cite the python notebook example

hbar=1
m=1
omega=1
N = 2015
a = 30.0
x = np.linspace(-a/2.,a/2.,N)
h = x[1]-x[0] # Should be equal to 2*np.pi/(N-1)
V = hbar*omega*x**2/2
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

order = 0
sign = 1

kk = int((N-1)/2)
#plt.plot(x[kk:],(psi[0]/np.sqrt(h))[kk:],label="$E_{}$={:3.1f}".format(0,En[0]))
psi_inp = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:] , kind='cubic')
psi_inp_lin = interp1d(x[kk:],(psi[order]/np.sqrt(h))[kk:])

plt.plot(x,sign*psi[order]/np.sqrt(h))
#plt.plot(x,np.exp(-x**2/2)/np.pi**(1/4))
plt.plot(x,0.1*V)
plt.show()

x_max = 5

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

#N_s = 300
## Generate complex moments?
#s1 = np.random.uniform(low = 1, high =1, size = N_s)
#s2 = np.random.uniform(low = -5*np.pi, high = 5*np.pi, size = N_s)
#s = [ 1.0 + t2*1j for t1,t2 in zip(s1,s2) ]
#q = vec_int(s)



def form(p,s): return 2**(p[0]*s + p[1])*gamma(p[2]*s)

p = np.random.random(size=(3))
print(p)

## First derive the s which make solving nice
s1 = - lambertw(-2**p[1] * p[0]* np.log(2))/p[0]/np.log(2)

print("S1:",s1)

## Gamma(3.56238228539090)= 3.5623...
s2 = 3.56238228539090/p[2]

print("S2:",s2)

q1 = special_int(s1)
print(q1)
q2 = special_int(s2)
print(q2)

def opt_fn(p,S,Q): return np.sum([(form(p,s)-q)**2 for s,q in zip(S,Q)])

## Solve for a new p
res = minimize(opt_fn,x0=p,args=([s1,s2],[q1,q2]))
print(res)


exit()




window_length = 10
s = [1.0 + 1j*t for t in np.linspace(-window_length,window_length,150)]
q = vec_int(s)
coords = np.imag(s)

#plt.plot(coords,np.real(q),label='real')
#plt.plot(coords,np.imag(q),label='imag')
## Interpolation
real_inp = interp1d(coords,np.real(q))
imag_inp = interp1d(coords,np.imag(q))
#real_inp  = lambda x : np.where( np.abs(x) <= window_length, real_inp_(np.clip(x,-window_length,window_length)), 0.0)
#imag_inp  = lambda x : np.where( np.abs(x) <= window_length, imag_inp_(np.clip(x,-window_length,window_length)), 0.0)



## Define the gamma wavelets
#real_gamma_inp = interp1d(np.imag(s2),gamma_r)
#imag_gamma_inp = interp1d(np.imag(s2),gamma_i)



plt.plot(coords,real_inp(coords))
plt.plot(coords,imag_inp(coords))
plt.show()



## A function to get the overlap integral as a function of a scale (a) and shift (b) parameter
## This treats the real and imaginary parts of the gamma function as wavelets
def real_overlap_integrand(x,a,b):
  s = 1.0 + 1j*(x-b)/a 
  return (np.real(gamma(s)) - real_inp(x))**2
def imag_overlap_integrand(x,a,b): 
  s = 1.0 + 1j*(x-b)/a 
  return (np.imag(gamma(s)) - real_inp(x) )**2
def real_gamma_overlap(a,b): return integrate.quad( real_overlap_integrand, -window_length, window_length, args=(a,b))[0]
def imag_gamma_overlap(a,b): return integrate.quad( imag_overlap_integrand, -window_length, window_length, args=(a,b))[0]

vec_real_gamma_overlap = np.frompyfunc(real_gamma_overlap,2,1)
vec_imag_gamma_overlap = np.frompyfunc(imag_gamma_overlap,2,1)

## For a variety of a and b parameters determine the best overlap
#N_a = 30
#N_b = 30
#a = np.linspace(0.5,10,N_a)
#b = np.linspace(-5,5,N_b)
#params_combinations = np.reshape([ [ [aa,bb] for bb in b ] for aa in a ],(N_a*N_b,2))

#pp = np.transpose(params_combinations)
#real_overlaps = vec_real_gamma_overlap(pp[0],pp[1])
#imag_overlaps = vec_imag_gamma_overlap(pp[0],pp[1])

from scipy.optimize import minimize
def opt_overlap(a): return vec_imag_gamma_overlap(*a) + vec_real_gamma_overlap(*a)

res = minimize(opt_overlap, x0=[1,0])
print(res)
exit()



if(True):
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(pp[0], pp[1], real_overlaps, c=real_overlaps, cmap='Reds');
  ax.scatter3D(pp[0], pp[1], imag_overlaps, c=imag_overlaps, cmap='Greens');
  ax.set_xlabel('a')
  ax.set_ylabel('b')
  ax.set_zlabel('Overlap')
  plt.show()

exit()


plt.plot(np.imag(s),ggg_r,label="gamma_real")
plt.plot(np.imag(s),ggg_i,label="gamma_imag")
plt.legend()
plt.show()

exit()

from scipy.optimize import fsolve


am = np.argmin(np.real(q))
print(am)
print(s[am])

##Look for a zero along the complex axis
def real_int(s): return np.real(vec_int(1+s*1j))

print(np.imag(s[am]))
res = fsolve(real_int,x0=np.imag(s[am]))
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


np.save("s_values_H",s)
np.save("moments_H",q)
np.save("logmoments_H",np.log(q))

