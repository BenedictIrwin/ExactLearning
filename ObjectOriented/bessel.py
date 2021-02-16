from mpmath import *
mp.dps = 15; mp.pretty = True

import numpy as np

S = [1+1j, 1/2 -1j, 1, 3/2, 0.1 + 0.1j, 2-1j/2, 1/2+2j]

res = []
order = 5

for s in S:
  f = lambda x : besselj(s,x)**order
  a = quadosc(f,[0,inf], omega = 3)
  #a = quad(f,[0,inf])
  res.append(a)
#print([gamma(1/6)*gamma(1/6 +s/2)/2**(5/3)/3**(1/2)/pi**(3/2)/gamma(5/6+s/2) for s in S])

res = np.array([complex(q) for q in res])

## Save the results
np.save("moments_BesselJ{}".format(order),res)
np.save("s_values_BesselJ{}".format(order),np.array(S))
np.save("real_error_BesselJ{}".format(order),np.array([1e-10 for s in S]))
np.save("imag_error_BesselJ{}".format(order),np.array([1e-10 for s in S]))

