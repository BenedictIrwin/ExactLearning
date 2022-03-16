import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import k0

## Randomly sample points from k0
## Do this according to the relative "density"
N = 200000
x = sorted(np.random.uniform(low=0,high=14,size=N)) ## draw 100 points between 1 and 2
values = k0(x)
sum_values = np.sum(values)
probs = [ xx/sum_values for xx in values ]
## Randomly select samples from values based on the distribution
d = np.random.choice(x,N,p=probs,replace=True)

#d= np.abs(np.random.normal(loc=0, scale = 1, size = N))

if(True):
  plt.hist(d,bins=100, normed=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()

## Generate complex moments?
s1 = np.random.uniform(low = 1, high =3, size = 1000)
s2 = np.random.uniform(low = -np.pi, high =np.pi, size = 1000)
s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
q = [np.mean(np.power(d,ss-1)) for ss in s]


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


np.save("s_values_bessel",s)
np.save("logmoments_bessel",np.log(q))

