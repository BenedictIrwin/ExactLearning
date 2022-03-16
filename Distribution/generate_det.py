import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


D  = np.random.random(size=[10000000,2,2])
D = [np.abs(np.linalg.det(d)) for d in D]
print(len(D))
if(True):
  plt.hist(D,bins=100, density=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()


## Generate complex moments?
s1 = np.random.uniform(low = 1, high =4, size = 1000)
s2 = np.random.uniform(low = -np.pi, high =np.pi, size = 1000)
s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
q = [np.mean(np.power(D,ss-1)) for ss in s]


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


np.save("s_values_det",s)
np.save("logmoments_det",np.log(q))
np.save("moments_det",q)

