import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

N = 100
means = []

R=10000

for i in range(R):
  x = np.random.uniform(size=(N,2))
  xx = np.array([1,0])
  yy = np.array([0,1])
  data = np.array([ [x + i*xx + j*yy for j in range(-1,2)] for i in range(-1,2) ])
  data = np.reshape(data, (3*3*N,2))
  
  print("Assembling Tree")
  
  tree = KDTree(data)
  res = tree.query(x,k=2)
  
  mean = [ i[1] for i in res[0]]
  means.append(np.array(mean))
  #print(np.mean(mean))
  #print("a~",np.pi*np.mean(mean)/2)

means = np.reshape(np.array(means),(R*N))

plt.hist(means,bins=500,density=True)

#a = 1/np.sqrt(4*N)
#x = np.linspace(np.amin(mean),np.amax(mean),100)
#y = 4/np.sqrt(np.pi)/a*(x/a)**2 * np.exp(-(x/a)**2)
#plt.plot(x,y)
plt.show()

## Sample
real_part = np.random.uniform(low=1,high=4,size=100)
imag_part = np.random.uniform(low=-np.pi,high=np.pi,size=100)
s = np.array([ r + 1j*i for r,i in zip(real_part,imag_part)])
moment = np.mean([np.power(means,ss) for ss in s],axis=1)
tag = "NNK1"
np.save("s_values_{}".format(tag),s)
np.save("moments_{}".format(tag),moment)
np.save("real_error_{}".format(tag),np.array([1e-15 for ss in s]))
np.save("imag_error_{}".format(tag),np.array([1e-15 for ss in s]))
