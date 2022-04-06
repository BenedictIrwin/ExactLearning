
## Simple hardcoded example

#phi(s) = gamma
from scipy.special import gamma
import numpy as np

def phi(s): return 2**(s/2-1) * gamma(s/2)

s = np.array([1.15,1.1,1.2,1.3,2.1,3.1])
v = np.array([ phi(s-2), phi(s), phi(s+2)])
p = np.array([s**0, s**1, s**2])

print(s,s.shape)
print(v,v.shape)
print(p,p.shape)

M_init = np.random.uniform(-1,1, size = 3*3)


def loss(M):
  M = M.reshape((3,3))
  results = np.einsum("jn,jk->nk",v,M)
  results = np.einsum("nk,kn->n",results,p)
  return np.mean(results**2)

## Also consider scaling output so a result is one?


loss(M_init)

from scipy.optimize import minimize

res = minimize(loss, x0 = M_init, method = "Nelder-Mead", tol = 1e-25)
print(res)

sqrtdet = np.sqrt(np.abs(np.linalg.det(res.x.reshape(3,3))))

print(res.x.reshape((3,3))/sqrtdet)
print(res.x.reshape((3,3)))
#print(np.array([[1,-1],[1,0]]))

M_true = np.array([2,-3,1,1,0,0,-1,0,0])
print(loss(M_true))
print(loss(M_init))

