import numpy as np

x0 = np.random.exponential(scale=1,size=10000)
x1 = np.random.exponential(scale=1,size=1000)
x2 = np.random.exponential(scale=1,size=1000)

for i in range(len(x0)):
  #print("{},{},{}".format(x0[i],x1[i],x2[i]))
  #print("{},{}".format(x0[i],x1[i]))
  print("{}".format(x0[i]))
