from scipy.special import jn
import numpy as np
from matplotlib import pyplot as plt

N = 10000
min = 0.1
max = 40.0

h = (max-min)/N
x = np.linspace(min,max,N)

# Exact answers ##########
y = jn(0,x)
dy = -jn(1,x)
ddy = -jn(0,x) + jn(1,x)/x

# Approximate gradients ###############
eo = 2
ndy = np.gradient(y, edge_order = eo)/h
nddy = np.gradient(ndy, edge_order = eo)/h

# Cut the edges off the function
x = x[2:-2]
y = y[2:-2]
ndy = ndy[2:-2]
nddy = nddy[2:-2]
dy = dy[2:-2]
ddy = ddy[2:-2]


plt.plot(x,y,label='y(x)')
plt.plot(x,dy,label='dy(x)')
plt.plot(x,ddy,label='ddy(x)')
plt.plot(x,ndy,label='ndy(x)')
plt.plot(x,nddy,label='nddy(x)')
plt.legend()
plt.show()

plt.plot(x, x**2 * ddy + x * dy + x**2 * y)
plt.plot(x, x**2 * nddy + x * ndy + x**2 * y)
plt.show()


# 3 x t
derivs = np.array([y,ndy,nddy])

# 3 x t
space = np.array([x**0,x,x**2])

# 3 x 3
# Note this is only the n=0 matrix
Bessel = np.array([[0,0,1],[0,1,0],[0,0,1]])

# general_Bessel = np.array([[-a**2,0,1],[0,1,0],[0,0,1]])


answer = np.einsum("it,ij->jt",derivs,Bessel)
answer = np.einsum("jt,jt->t",answer,space)
print(answer)
print(np.sum(answer))
print("Error per sample: ",np.sum(answer)/N)

