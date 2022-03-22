
from mpmath import psi
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9):
  S = array([(1.3249851465108757+0.6377964194732235j), (3.0971456270892688+1.215502041762794j), (3.370170461184666+2.7457744690790538j), (4.202458092216693-1.952538391913905j), (4.781273136950069+2.759125160468188j), (2.521298661318751-1.333703460781513j), (1.085679138296996-2.974670971702598j), (4.599092584710192+0.5068929083757183j), (1.1046031106504022-3.1275668070219815j), (2.203835032153134-0.1684013080857043j)])
  ret = 0
  ret += 0
  ret += 0
  ret += p3**2*array([complex(psi(1,p2 + ss*p3)) for ss in S])
  ret += p5**2*array([complex(psi(1,p4 + ss*p5)) for ss in S])
  ret += -p7**2*array([complex(psi(1,p6 + ss*p7)) for ss in S])
  ret += -p9**2*array([complex(psi(1,p8 + ss*p9)) for ss in S])
  return ret
logderivative2 = frompyfunc(fp,10,1)
