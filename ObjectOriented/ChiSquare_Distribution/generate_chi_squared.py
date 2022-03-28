import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

#import scipy.integrate as integrate

#def real_integrand(x,s): return np.real(x**(s-1)*np.exp(-x**2-x))
#def imag_integrand(x,s): return np.imag(x**(s-1)*np.exp(-x**2-x))
#def special_int(s):  
#  r = integrate.quad(real_integrand, 0, np.inf, args=(s))
#  i = integrate.quad(imag_integrand, 0, np.inf, args=(s))
#  #return (r[0]+ 1j*i[0],r[1:],i[1:])  
#  return r[0]+ 1j*i[0], r[1], i[1]

#vec_int = np.vectorize(special_int)

from scipy.special import gamma
keyword = "ChiSquare_Distribution"

size = 100000

k = 4
lengths = np.random.chisquare(k, size = size)

## Show a histogram
plt.hist(lengths, bins = 100, density = True)

xx = np.linspace(0,20,100)
ff = (1/2)**(k/2)/gamma(k/2) *xx**(k/2-1) * np.exp(-xx/2)
plt.plot(xx,ff,'k-')

#from scipy.stats import chi
#xx = np.linspace(0,5,100)
#yy = chi.pdf(xx, 10)
#plt.plot(xx,yy,'-k')

plt.show()

#exit()



## Generate reail and imaginary part complex moments
s_size = 5
s1 = np.random.uniform(low = 1, high = 7, size = s_size)
s2 = np.random.uniform(low = -1*np.pi, high = 1*np.pi, size = s_size)
s = np.expand_dims(s1 + s2*1j, axis = 0)


s = 1+1j*np.linspace(-3*np.pi,3*np.pi,300)

print(s)

#s = np.array([1+0j,2+0j,3+0j,4+0j, 1.5+1j, 1.5-1j, 2.5+1.5j, 2.5-1.5j])

#q, qre, qie = vec_int(s)



## For each moment, get the expectation of s-1 for Mellin Transform
q = np.mean(np.power(np.expand_dims(lengths,1),s-1),axis=0)

power = q

def phi(s): return 2**(s-1)*gamma(s+1)

ss = np.imag(s)

#ss = s[0]

#print(q)
#print(q_true)
#exit()

## Also get the expectation of E[logX X^(s-1)] which is the derivative of the fingerprint.

dq = np.mean( np.expand_dims(np.log(lengths), axis = 1) * np.power(np.expand_dims(lengths,1),s-1),axis=0)
ddq = np.mean( np.expand_dims(np.log(lengths)**2, axis = 1) * np.power(np.expand_dims(lengths,1),s-1),axis=0)
#s = s[0]

def maxx_f(s): return np.amax(lengths)**(s-1)/size
def minn_f(s): return np.amin(lengths)**(s-1)/size
def example_f(s,x): return x**(s-1)/size

plt.plot(ss,np.abs(phi(s)),'r-',label="Theoretical phi(s)")
plt.plot(ss,np.abs(power),'k:',label = "Numerical E[x^(s-1)]")
plt.plot(ss,np.abs(maxx_f(s)),'b:',label = "x_max^(s-1)")
plt.plot(ss,np.abs(np.exp(np.log(power) - np.amax([np.log(minn_f(s)),np.log(maxx_f(s))],axis = 0))/size),'k-',label = "Error Profile")
plt.legend()
plt.show()
plt.plot(ss,np.real(np.log(np.abs(phi(s)))),'r-',label="Theoretical logphi(s)")
plt.plot(ss,np.log(np.abs(power)),'k:',label = "Numerical logE[x^(s-1)]")
plt.plot(ss,np.log(np.abs(maxx_f(s))),'b:',label = "log(x_max^(s-1))")
plt.plot(ss,np.log(np.abs(minn_f(s))),'b:',label = "log(x_min^(s-1))")
plt.plot(ss,np.abs(np.log(power) - np.amax([np.log(minn_f(s)),np.log(maxx_f(s))],axis = 0)),'k-',label = "log(Error Profile)")

#for y in np.linspace(np.amin(sample),np.amin(sample)*3,100):
#  plt.plot(ss,np.log(example_f(ss,y)),'g:')
plt.legend()
plt.show()
plt.plot(ss,np.abs(phi(s)-power),'r-',label="Theoretical - Numerical")
plt.plot(ss,0.02*np.abs(np.exp(np.log(power) - np.amax([np.log(minn_f(s)),np.log(maxx_f(s))],axis = 0))/size),'k-',label = "Error Profile")
plt.plot(ss,[0 for q in ss],'k:')
plt.legend()
plt.show()

exit()



## Get the ratio...
def plot_three_dim(q):
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

if(False):
  plot_three_dim(q)
  plot_three_dim(dq)
  plot_three_dim(dq/q)



## Save out exact learning data (complex moments)
np.save("s_values_{}".format(keyword),s)
np.save("moments_{}".format(keyword),q)
np.save("logmoments_{}".format(keyword),np.log(q))
np.save("derivative_{}".format(keyword),dq)
np.save("logderivative_{}".format(keyword),dq/q)
np.save("logderivative2_{}".format(keyword),ddq/q)




#np.save("real_error_{}".format(keyword),qre)
#np.save("imag_error_{}".format(keyword),qie)
