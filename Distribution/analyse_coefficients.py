import numpy as np
from matplotlib import pyplot as plt

## Read files KNN
losses1 = np.load("KNN_losses_1.npy")
params1 = np.load("KNN_params_1.npy")
losses2 = np.load("KNN_losses_2.npy")
params2 = np.load("KNN_params_2.npy")

l = np.concatenate((losses1,losses2))
p = np.concatenate((params1,params2))

w = [1.0/ll for ll in l]

count = 0
for param in np.transpose(p):
  if(count in [1,2]): param = np.abs(param)
  avg = np.average(param, weights = w)
  plt.hist(param, bins = 100, label = "raw", density = True)
  a,b,c = plt.hist(param, bins = 100, weights = w, label ="weighted", density = True)
  best = np.argmax(a)
  print(b[best], b[best+1])
  plt.legend()
  plt.show()
  count+=1


