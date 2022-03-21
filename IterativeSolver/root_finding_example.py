
from numpy import log, exp
from scipy.special import gamma, digamma
import numpy as np

## Define a secret function 'moments'
def phi(s, v):
  return v[0]*v[1]**s*gamma(v[2] + v[3]*s)/gamma(v[4] + v[5]*s)

## Logarithmic derivative of it
def logdivphi(s,v):
  return log(v[1]) + v[3]*digamma(v[2] + v[3]*s) - v[5]*digamma(v[4] + v[5]*s)

## EulerGamma
g = 0.57721566490153286060651209008240243104216

c = np.random.uniform(0,1,size=[6])  ## initial
v = np.random.uniform(0,1,size=[6])  ## hidden actual

from scipy.optimize import root

#def fun(x,c):
#  return np.real(gamma(x[0] + x[1]*1j) - c), np.imag(gamma(x[0] + x[1]*1j) - c)

def fun(x,c):
  return gamma(x) - c

## Grad is the option to extract the b term in a+ b*s 
## Default is to extract the a term
def get_root(p0,p1, grad = False):
  success = False
  it = 0
  x0 = 10
  k = -1/1.5
  while(not success):
    k = -1.5*k
    if(grad): r = root(fun, x0=x0, args = (k*p1))
    else: r = root(fun, x0=x0, args = (k*p0))
    #print(r)
    success = r.success
    it+=1
  return (r.x[0]-p0)/p1, k


print("start c",c)
for i in range(10):
  ## Initialise a space
  c_new = np.zeros(len(c))  
  
  ## Critical Points
  p0 = phi(0, v)
  p1 = phi(1, v)

  c_new[0] = p0*gamma(c[4])/gamma(c[2])
  c_new[1] = p1*gamma(c[4]+c[5])/gamma(c[2]+c[3])/c[0]

  #pp,qq = -10, 0.91
  #pp,qq = np.pi, exp(1)
  #a, k = get_root(pp,qq)

  ## Workign to some extent
  #print(a,k)
  #print("G(a+bs)",gamma(pp + a*qq))
  #print("G(a+bs)/k",gamma(pp + a*qq)/k)
  #print("a,b", pp, qq)


  ## Try a new function that makes
  #Gamma(a + b * root) = a * k
  ## With k being a number that makes a*k large and positive..
  ## However we are not apriori going to know what a is....
  ## But we will pretend we do as we iterate... so k is also likely to change as the iteration goes on... 

  c = np.nan_to_num(c, nan = 1)


  root_2, k_2 = get_root(c[2],c[3])
  root_3, k_3 = get_root(c[2],c[3], True)
  root_4, k_4 = get_root(c[4],c[5])
  root_5, k_5 = get_root(c[4],c[5], True)


  #print("root, k",root_2, k_2)
  #print("c2, gamma/k",c[2],gamma(c[2] + root_2 * c[3])/k_2)
  #print(c[2],"->",c_new[2])
  

  #print("root3, k3",root_3, k_3)
  #print("c3, gamma/k3",c[3],gamma(c[2] + root_3 * c[3])/k_3)

  ## Upper gammas
  c_new[2] = gamma(c[4] + c[5]*root_2)*phi(root_2,v)/k_2/c[0]/c[1]**root_2  
  c_new[3] = gamma(c[4] + c[5]*root_3)*phi(root_3,v)/k_3/c[0]/c[1]**root_3 


  ## Note these are lower gammas
  c_new[4] = gamma(c[2] + c[3]*root_4)*c[0]*c[1]**root_4 * k_4/phi(root_4,v)
  c_new[5] = gamma(c[2] + c[3]*root_5)*c[0]*c[1]**root_5 * k_5/phi(root_5,v)

  #print(c,"->", c_new)
  #print(c)
  #for i in range(len(c_new)):
  #  if(abs(c_new[i]) > 100): c_new[i] = np.random.uniform(-1,1)

  #c = 0.99*c + 0.01*c_new

  c[0] = c_new[0]
  c[1] = c_new[1]
  c[2] = c_new[2]


  print(c)



print("c",c)
print("v",v)





