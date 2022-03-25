
from numpy import log
from scipy.special import digamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6):
  S = array([(2.1486417085817924+1.487770582985024j), (3.0779688396777107+2.3546260992104937j), (3.969093569902811+2.361265998390386j), (1.048388892463171-3.1110928674353264j), (4.733445650714383+3.0276741242998604j)])
  ret = 0
  ret += 0
  ret += log(p1**2)
  ret += digamma(p2 + S)
  ret += p4*digamma(p3 + S*p4)
  ret += -p6*digamma(p5 + S*p6)
  return ret
logderivative = frompyfunc(fp,7,1)
