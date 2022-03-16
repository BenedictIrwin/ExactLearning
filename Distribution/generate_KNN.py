import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree


D = []
R = 1000
for reps in range(R):
  N = 1000
  orig_pairs = np.transpose([np.random.random(N),np.random.random(N)])
  shift_pairs = np.array([[ orig_pairs + np.array([xs,ys]) for xs in range(-1,2)] for ys in range(-1,2)])
  shift_pairs = np.reshape(shift_pairs,[N*3*3,2])
  
  if(False):
    plt.scatter([x[0] for x in shift_pairs],[x[1] for x in shift_pairs])
    plt.scatter([x[0] for x in orig_pairs],[x[1] for x in orig_pairs])
    plt.show()
  
  tree = KDTree(shift_pairs)
  
  res = tree.query(orig_pairs,k=3)
  d = [ x[2] for x in res[0]]
  D.append(d)

D = np.reshape(D,[N*R])

print(len(D))
if(True):
  plt.hist(D,bins=100, density=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()


## Generate complex moments?
s1 = np.random.uniform(low = -2, high =9, size = 2000)
s2 = np.random.uniform(low = -2*np.pi, high =2*np.pi, size = 2000)
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


np.save("s_values_NN1",s)
np.save("logmoments_NN1",np.log(q))
np.save("moments_NN1",q)

