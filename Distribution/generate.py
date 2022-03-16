import numpy as np
from matplotlib import pyplot as plt

N = 10000000
r = np.sqrt(np.random.random(N))
theta = np.random.random(N) * 2 * np.pi
x_circ = r*np.cos(theta)
y_circ = r*np.sin(theta)

x_square = np.random.random(N)
y_square = np.random.random(N)

d = np.sqrt( (x_circ-x_square)**2 + (y_circ-y_square)**2 )

if(False):
  plt.hist(d,bins=100, normed=True)
  plt.xlabel("Distance")
  plt.ylabel("Probability Density")
  plt.show()

s = np.linspace(1,6,200)

q = [np.mean(np.power(d,ss-1)) for ss in s]

if(False):
  plt.plot(s,q)
  plt.xlabel("s")
  plt.ylabel("$E[x^{s-1}]$")
  plt.show()

  plt.plot(s,np.log(q))
  plt.xlabel("s")
  plt.ylabel("$\log(E[x^{s-1}])$")
  plt.show()

np.save("s_values",s)
np.save("logmoments",np.log(q))

