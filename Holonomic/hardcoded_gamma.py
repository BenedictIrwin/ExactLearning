
## Simple hardcoded example

#phi(s) = gamma
from scipy.special import gamma
import numpy as np

s = np.array([1.15,1.1,1.2,1.3])
v = np.array([ gamma(s-1), gamma(s)])
p = np.array([s**0, s**1])

print(s,s.shape)
print(v,v.shape)
print(p,p.shape)

M = np.eye(2)

M_init = np.array([1,0,0,1])


def loss(M):
  M = M.reshape((2,2))
  results = np.einsum("jn,jk->nk",v,M)
  results = np.einsum("nk,kn->n",results,p)
  return np.mean(results**2)

## Also consider scaling output so a result is one?


loss(M_init)

from scipy.optimize import minimize

res = minimize(loss, x0 = M_init)
print(res)

sqrtdet = np.sqrt(np.linalg.det(res.x.reshape(2,2)))

print(res.x.reshape((2,2))/sqrtdet)



print(np.array([[1,-1],[1,0]]))


