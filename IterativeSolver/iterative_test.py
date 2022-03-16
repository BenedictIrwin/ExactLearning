
from numpy import log, exp
from scipy.special import gamma, digamma
import numpy as np

## Define a secret function 'moments'
def phi(s, v):
  return v[0]*v[1]**s*gamma(v[2] + v[3]*s)

def logdivphi(s,v):
  return log(v[1]) + v[3]*digamma(v[2] + v[3]*s)

## Answer = [2,2,1,0.5]

#randomly initialise
c1 = 2
c2 = 2
c3 = 1.05
c4 = 0.5

## EulerGamma
g = 0.57721566490153286060651209008240243104216

c1,c2,c3,c4 = np.random.uniform(0,1,size=[4])
v = np.random.uniform(0,1,size=[4])

print(c1,c2,c3,c4)

for i in range(1000):
  ## Solve
  p0 = phi(0, v)
  p1 = phi(1, v)

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





