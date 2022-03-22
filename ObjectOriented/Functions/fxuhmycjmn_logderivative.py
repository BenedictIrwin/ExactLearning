
from numpy import log
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([3.75327895+0.98377319j, 2.26806717+1.88100072j,
       4.47028628-2.09062136j, 3.46427733+1.73331923j,
       1.92512322-0.25598092j, 2.46506712-0.16224504j,
       2.42862054+2.91408707j, 1.59380343+1.02864922j,
       1.28712215+2.24043744j, 2.27946732+2.20656273j])])
  ret = 0
  ret += 0
  ret += log(p1**2)
  ret += p3*digamma(p2 + S*p3)
  return ret
logderivative = frompyfunc(fp,4,1)
