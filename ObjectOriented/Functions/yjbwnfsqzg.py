from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([(2.3509518030149676+1.165671659511788j), (3.7371941526023784-2.9987750305526273j), (5.506077880670518+2.389747040896948j), (2.707852733570908+1.4901387483434831j), (1.1362068798157368+1.645097505880904j)])
  ret = 0
  ret += log(p0**2)
  ret += loggamma(p1 + S)
  ret += -loggamma(p2 + S)
  return ret
fingerprint = frompyfunc(fp,3,1)
