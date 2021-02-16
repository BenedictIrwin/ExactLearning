import numpy as np
from matplotlib import pyplot as plt

means = []
varss = []
expvs = []

for i in range(100000):
  exp_data = np.random.exponential(scale=1.0,size=[50])**2
  d2=exp_data**2
  means.append(np.mean(exp_data))
  varss.append(np.std(exp_data)**2)
  expvs.append(np.mean(d2)-np.mean(exp_data)**2)

plt.hist(means,bins=100,alpha=0.2,color='blue',density=True)
plt.hist(varss,bins=100,alpha=0.2,color='red',density=True)
plt.hist(expvs,bins=100,alpha=0.2,color='green',density=True)
plt.show()

