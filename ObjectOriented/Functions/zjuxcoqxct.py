from numpy import log
from scipy.special import loggamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([(1+0j), (2+0j), (3+0j), (4+0j), (1.5+1j), (1.5-1j), (2.5+1.5j), (2.5-1.5j)])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
