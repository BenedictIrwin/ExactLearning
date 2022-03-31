import matplotlib.pyplot as plt
from scipy.special import loggamma, gammaln, gammasgn, gamma
import numpy as np

x = np.linspace(-4,8,500)


plt.title("Comparison of Activation Functions")
plt.plot(x,np.tanh(x),label="tanh(x)")
plt.plot(x,gammaln(x),label="log|$\Gamma$(x)|")
plt.plot(x,x,label="x")
plt.xlabel("x")
plt.legend()
plt.grid()

plt.show()



