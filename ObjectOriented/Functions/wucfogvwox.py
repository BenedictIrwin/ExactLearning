from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([2.45390573-1.51279977j, 1.26176703+1.71303757j,
       4.27950813-2.03810251j, 2.09589842+1.18454068j,
       4.43245432-2.13450082j, 1.26297508+2.83285265j,
       2.73954829+1.41861839j, 2.41070426-2.03582681j,
       1.31168445+1.84082893j, 3.26513661+1.22611988j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
