import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

N = 1000

s_samples = [1.0+i/10.0 for i in range(10)]

num_repeats = 10000
for s in s_samples:
  list_of_means = []
  list_of_stds = []
  list_of_sqdiff = []
  for rep in range(num_repeats):
    exp_samps = np.random.exponential(scale=1.0,size=N)
    data = exp_samps**s
    list_of_means.append(np.mean(data))
    list_of_stds.append(np.std(data))
    sqdiff = (np.mean(data)-gamma(s+1))**2
    list_of_sqdiff.append(sqdiff)
  print("{},{},{}".format(s,np.mean(list_of_sqdiff),gamma(s+1)))
