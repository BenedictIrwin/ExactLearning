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

keyword = "Disk_Line_Picking"

size = 1000000

## Generate uniformly smapled points on the unit disk
random_r1 = np.random.uniform(0,1,size)
random_theta1 = np.random.uniform(0,2*np.pi,size)
random_r2 = np.random.uniform(0,1,size)
random_theta2 = np.random.uniform(0,2*np.pi,size)

## Convert to cartesian coordinates
random_x1 = np.sqrt(random_r1)*np.cos(random_theta1)
random_y1 = np.sqrt(random_r1)*np.sin(random_theta1)
random_x2 = np.sqrt(random_r2)*np.cos(random_theta2)
random_y2 = np.sqrt(random_r2)*np.sin(random_theta2)

## Get the lengths of random lines
lengths = np.sqrt( (random_x1 - random_x2)**2 + (random_y1 - random_y2)**2  )

## Show a histogram
plt.hist(lengths, bins = 100, density = True)

xx = np.linspace(0,2,100)
ff = 4*xx/np.pi * np.arccos(xx/2) - 2 * xx**2/np.pi * np.sqrt(1 - xx**2/4) 
plt.plot(xx,ff,'k-')

#from scipy.stats import chi
#xx = np.linspace(0,5,100)
#yy = chi.pdf(xx, 10)
#plt.plot(xx,yy,'-k')

plt.show()

#exit()



## Generate real and imaginary part complex moments
s_size = 10
s1 = np.random.uniform(low = 1, high = 7, size = s_size)
s2 = np.random.uniform(low = -1*np.pi, high = 1*np.pi, size = s_size)
s = np.expand_dims(s1 + s2*1j, axis = 0)
print(s)

#s = np.array([1+0j,2+0j,3+0j,4+0j, 1.5+1j, 1.5-1j, 2.5+1.5j, 2.5-1.5j])

#q, qre, qie = vec_int(s)



## For each moment, get the expectation of s-1 for Mellin Transform
q = np.mean(np.power(np.expand_dims(lengths,1),s-1),axis=0)

from scipy.special import gamma

#q_true = 4 * 2**s * gamma(1 + s/2) * gamma(1 + s)/ np.sqrt(np.pi) /gamma(2+s)/gamma(5/2 + s/2)


#print(q)
#print(q_true)
#exit()

## Also get the expectation of E[logX X^(s-1)] which is the derivative of the fingerprint.

dq = np.mean( np.expand_dims(np.log(lengths), axis = 1) * np.power(np.expand_dims(lengths,1),s-1),axis=0)
ddq = np.mean( np.expand_dims(np.log(lengths)**2, axis = 1) * np.power(np.expand_dims(lengths,1),s-1),axis=0)
dddq = np.mean( np.expand_dims(np.log(lengths)**3, axis = 1) * np.power(np.expand_dims(lengths,1),s-1),axis=0)
s = s[0]

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
np.save("logderivative3_{}".format(keyword),dddq/q)




#np.save("real_error_{}".format(keyword),qre)
#np.save("imag_error_{}".format(keyword),qie)
