
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6):
  S = array([(2.1486417085817924+1.487770582985024j), (3.0779688396777107+2.3546260992104937j), (3.969093569902811+2.361265998390386j), (1.048388892463171-3.1110928674353264j), (4.733445650714383+3.0276741242998604j)])
  ret = 0
  ret += 0
  ret += digamma(p1 + S)
  ret += p3*digamma(p2 + S*p3)
  ret += -p5*digamma(p4 + S*p5)
  ret += -digamma(p6 + S)
  return ret
logderivative = frompyfunc(fp,7,1)
