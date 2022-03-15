from numpy import log
from scipy.special import loggamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([1.609058  +1.34854442j, 3.4633344 -2.75912716j,
       1.44772495-2.57037062j, 3.53092859-3.06117092j,
       4.90438126-1.40274682j, 3.28730605-2.61239124j,
       2.18078472+1.49314937j, 3.79698537+2.05262343j,
       4.83963649-0.5410193j , 3.28570095+0.19013669j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
