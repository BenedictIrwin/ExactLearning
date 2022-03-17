
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

def fun(x,c):
  return np.real(gamma(x[0] + x[1]*1j) - c), np.imag(gamma(x[0] + x[1]*1j) - c)

def fun(x,c):
  return gamma(x) - c

def get_root(p0,p1):
  success = False
  it = 0
  while(not success):
    x0 = np.random.uniform(-10,50) 
    if(it==0): x0 =10
    r = root(fun, x0=x0, args = (p0))
    print(r)
    success = r.success
    it+=1
  return (r.x[0]-p0)/p1

for i in range(1000):
  ## Initialise a space
  c_new = np.zeros(len(c))  
  
  ## Critical Points
  p0 = phi(0, v)
  p1 = phi(1, v)

  c_new[0] = p0*gamma(c[4])/gamma(c[2])
  c_new[1] = p1*gamma(c[4]+c[5])/gamma(c[2]+c[3])/c[0]

  pp,qq = 0.74, 0.91
  #pp,qq = np.pi, exp(1)
  a = get_root(pp,qq)
  print(a)
  print(gamma(pp + a*qq))

  exit()

  root_2 = get_root(c[2],c[3])
  print(root_2)
  print(c[2],gamma(c[2] + root_2 * c[3]))
  exit()

  #c_new[2] =  
  #c_new[3] =  ## derivative?
  #c_new[4] =
  #c_new[5] =  ## derivative?


    
  ## Solve

  dp0 = logdivphi(0, v)

  t1 = phi((1-c3)/c4, v)
  t2 = phi((2-c3)/c4, v)
  ##log_term = np.log(t2/t1) / np.log(c2) 

  th1 = logdivphi((1-c3)/c4, v)
  th2 = logdivphi((2-c3)/c4, v)

  c3_new = t1*phi(1/c4, v)/p0/t2


  #c2_new = t2**c4/t1**c4
  #c2_new = p1/c1/gamma(c3+c4)
  #c2_new = exp(dp0 - c4*digamma(c3))
  c1_new = p0/gamma(c3)
  #c4_new = 1/log_term
  #c4_new = (dp0 - log(c2))/digamma(c3)
  #c4_new = (dp0 - log(c2_new))/digamma(c3)# + 0.05*(th2-th1)/(1-2*g)
  c4_new = th2 - th1
  c2_new = exp((th2 + th1 - c4_new*(1 - 2*g))/2)

  c1 = c1_new
  c2 = c2_new
  c3 = c3_new
  c4 = c4_new
  
  print(c1,c2,c3,c4)

print(v)

exit()

M = [[1,-digamma(c3)],[2,1-2*g]]
t = [dp0,th1+th2]

sol = np.linalg.inv(M)@t
print(sol)
print(exp(sol))

print(digamma(1))
print(g)

print(logdivphi((2-1)/0.5))
print(logdivphi((1-1)/0.5))
print(logdivphi((2-1)/0.5) - logdivphi((1-1)/0.5))


print(logdivphi((2-1)/0.5) + logdivphi((1-1)/0.5))
print(2*log(2) + 0.5*(1 - 2*g))





