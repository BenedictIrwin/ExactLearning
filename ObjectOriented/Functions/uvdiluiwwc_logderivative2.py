
from mpmath import psi
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9):
  S = array([(1.5449499493225458-0.875844681496571j), (2.188288016964692+2.806249518438454j), (4.043360573889501-2.8295027582877976j), (1.4171562240067845-2.1415649971114803j), (3.4666928262146595+3.1203652524117382j), (1.9268284010397734+1.1245784795652334j), (4.912487190158007+0.771389725873334j), (4.753624510306686-2.7053802484408647j), (1.0095174781675298+1.4319213725064435j), (1.316656124138257-2.2257255267094216j)])
  ret = 0
  ret += 0
  ret += 0
  ret += p3**2*array([complex(psi(1,p2 + ss*p3)) for ss in S])
  ret += p5**2*array([complex(psi(1,p4 + ss*p5)) for ss in S])
  ret += -p7**2*array([complex(psi(1,p6 + ss*p7)) for ss in S])
  ret += -p9**2*array([complex(psi(1,p8 + ss*p9)) for ss in S])
  return ret
logderivative2 = frompyfunc(fp,10,1)
