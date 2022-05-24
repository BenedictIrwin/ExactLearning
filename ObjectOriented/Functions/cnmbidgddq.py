from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([(6.519828860631247+2.1688706280290493j), (2.143262256097679-2.2581696754311102j), (3.5203571533018456+0.0622303975025309j), (6.623183702442857+2.0843551976366017j), (2.8809563311361317+2.722372479662739j)])
  ret = 0
  ret += log(p0**2)
  ret += loggamma(p1 + S)
  ret += -loggamma(p2 + S)
  return ret
fingerprint = frompyfunc(fp,3,1)
