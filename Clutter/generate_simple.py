import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


N = 100000

d= np.abs(np.random.normal(loc=0, scale = 1, size = N))

if(True):
  plt.hist(d,bins=100, normed=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()

## Generate complex moments?
s1 = np.random.uniform(low = 1, high =7, size = 400)
s2 = np.random.uniform(low = -4, high =4, size = 400)
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


np.save("s_values_simple",s)
np.save("logmoments_simple",np.log(q))

