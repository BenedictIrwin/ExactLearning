
from AdvancedFunctions import *
from numpy import array, frompyfunc
def fp(p0,p1,p2):
  S = array([(6.4174093680277045-1.797701018951581j), (5.980851012582712-2.9110778334390166j), (4.659279974992842-2.669337684661386j), (4.635398549744459-1.119942674147587j), (1.317253517511289+0.5441072636027418j)])
  ret = 0
  ret += 0
  ret += 0
  ret += np.trigamma(p2 + S)
  return ret
logderivative2 = frompyfunc(fp,3,1)
