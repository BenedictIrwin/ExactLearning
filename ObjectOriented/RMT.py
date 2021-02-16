import numpy as np

from matplotlib import pyplot as plt

## Best eigs
def eigs(M):
 eig, _ = np.linalg.eig(M)
 return np.amax(np.abs(eig))

#2x2 matrix
M = np.random.normal(size=(1000000,2,2))

E = [ eigs(q) for q in M]

print(np.amin(E))
print(np.amax(E))

plt.hist(E,bins = 200)

real_part = np.random.uniform(low=1,high=4,size=100)
imag_part = np.random.uniform(low=-np.pi,high=np.pi,size=100)


s = np.array([ r + 1j*i for r,i in zip(real_part,imag_part)])
moment = np.mean([np.power(E,ss) for ss in s],axis=1)

np.save("s_values_RMT",s)
np.save("moments_RMT",moment)
np.save("real_error_RMT",np.array([1e-15 for ss in s]))
np.save("imag_error_RMT",np.array([1e-15 for ss in s]))
