
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
  S = array([array([4.01652653-1.33571415j, 4.07414274-0.56844817j,
       1.76775959-1.47704689j, 3.29654365+1.46133632j,
       1.80671431+0.52507335j, 1.3699895 -1.78391995j,
       1.22131065-1.9216671j , 4.2004656 +1.2371831j ,
       3.9040192 +0.46054058j, 4.92557893+1.42942515j])])
  ret = 0
  ret += 0
  ret += p2*digamma(p1 + S*p2)
  ret += p4*digamma(p3 + S*p4)
  ret += -p6*digamma(p5 + S*p6)
  ret += -p8*digamma(p7 + S*p8)
  ret += -p10*digamma(p9 + S*p10)
  return ret
logderivative = frompyfunc(fp,11,1)
