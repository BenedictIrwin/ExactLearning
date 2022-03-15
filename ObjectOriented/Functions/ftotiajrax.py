from numpy import log
from scipy.special import loggamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([2.038588  -2.6053556j , 2.65733822+1.21002043j,
       1.39825951-2.99824264j, 2.19584816-0.60459726j,
       3.10491903+0.03158228j, 1.91926528-2.63540132j,
       3.99493744-2.41071149j, 3.01108922+0.07264158j,
       3.48395735+0.4945344j , 2.33114502-2.66264967j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
