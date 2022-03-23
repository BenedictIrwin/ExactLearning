
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([(1.620878541191576+1.124526651232462j), (6.27596065807341-1.6028221984675337j), (2.483429615859781+2.49986496665612j), (4.3757951962805235-0.2617572432616275j), (2.539312615377251-1.7702932934256894j)])
  ret = 0
  ret += 0
  ret += -digamma(p1 + S)
  ret += digamma(p2 + S)
  return ret
logderivative = frompyfunc(fp,3,1)
