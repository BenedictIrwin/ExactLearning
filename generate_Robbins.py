import numpy as np
from matplotlib import pyplot as plt

N = 10000000

x_1 = np.random.random(N)
y_1 = np.random.random(N)
z_1 = np.random.random(N)
x_2 = np.random.random(N)
y_2 = np.random.random(N)
z_2 = np.random.random(N)

d = np.sqrt( (x_1-x_2)**2 + (y_1-y_2)**2 + (z_1 - z_2)**2 )

if(True):
  plt.hist(d,bins=100, normed=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()

s = np.linspace(1,20,500)

q = [np.mean(np.power(d,ss-1)) for ss in s]

if(True):
  plt.plot(s,q)
  plt.xlabel("s")
  plt.ylabel("$E[x^{s-1}]$")
  plt.show()

  plt.plot(s,np.log(q))
  plt.xlabel("s")
  plt.ylabel("$\log(E[x^{s-1}])$")
  plt.show()

np.save("s_values_Robbins",s)
np.save("logmoments_Robbins",np.log(q))

