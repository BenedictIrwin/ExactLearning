from scipy.special import jn
from scipy.interpolate import CubicSpline
import numpy as np
from matplotlib import pyplot as plt

N = 10000
min = 0.1
max = 40.0

h = (max-min)/N
x = np.linspace(min,max,N)

order = 5

# Exact answers ##########
y = jn(order,x)
dy = (jn(order-1,x)-jn(order+1,x))/2.0
ddy = (order + order**2 - x**2) * jn(order, x) / x**2  - jn(order-1,x)/x

# Spline representation of function
cs = CubicSpline(x,y)
cy = cs(x)
cdy = cs(x,1)
dcs = CubicSpline(x,cdy)
cddy = dcs(x,1)

# Approximate gradients ###############
eo = 2
ndy = np.gradient(y, edge_order = eo)/h
nddy = np.gradient(ndy, edge_order = eo)/h

# Cut the edges off the function
clip_len = 10
x = x[clip_len:-clip_len]
y = y[clip_len:-clip_len]
ndy = ndy[clip_len:-clip_len]
nddy = nddy[clip_len:-clip_len]
dy = dy[clip_len:-clip_len]
ddy = ddy[clip_len:-clip_len]
cy = cy[clip_len:-clip_len]
cdy = cdy[clip_len:-clip_len]
cddy = cddy[clip_len:-clip_len]

plt.plot(x,y,label='y(x)')
plt.plot(x,dy,label='dy(x)')
plt.plot(x,ddy,label='ddy(x)')
plt.plot(x,ndy,label='ndy(x)')
plt.plot(x,nddy,label='nddy(x)')
plt.legend()
plt.show()

plt.title("Residuals of derivative vs. numerical")
plt.plot(x,dy-ndy,label='dy(x)-ndy(x)')
plt.plot(x,dy-cdy,label='dy(x)-cdy(x)')
plt.show()

plt.title("Residuals of second derivative vs. numerical")
plt.plot(x,ddy-nddy,label='ddy(x)-nddy(x)')
plt.plot(x,ddy-cddy,label='ddy(x)-cddy(x)')
plt.show()

plt.title("Annihilation of Function")
plt.plot(x, x**2 * ddy + x * dy + (x**2 - order**2) * y, label = 'Pure')
plt.plot(x, x**2 * cddy + x * cdy + (x**2 - order**2) * y, label = 'C-Spline')
plt.legend()
plt.show()


# 3 x t
derivs = np.array([y,ndy,nddy])

derivs = np.array([y,cdy,cddy])

# 3 x t
space = np.array([x**0,x,x**2])

# 3 x 3
Bessel = np.array([[-order**2,0,1],[0,1,0],[0,0,1]])

# general_Bessel = np.array([[-a**2,0,1],[0,1,0],[0,0,1]])


answer = np.einsum("it,ij->jt",derivs,Bessel)
answer = np.einsum("jt,jt->t",answer,space)
print(answer)
print(np.sum(np.abs(answer)))
print("Error per sample: ",np.sum(np.abs(answer))/N)

