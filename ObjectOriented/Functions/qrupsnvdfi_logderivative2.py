
from mpmath import psi
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
  S = array([array([4.01652653-1.33571415j, 4.07414274-0.56844817j,
       1.76775959-1.47704689j, 3.29654365+1.46133632j,
       1.80671431+0.52507335j, 1.3699895 -1.78391995j,
       1.22131065-1.9216671j , 4.2004656 +1.2371831j ,
       3.9040192 +0.46054058j, 4.92557893+1.42942515j])])
  ret = 0
  ret += 0
  ret += p2**2*array([complex(psi(1,p1 + ss*p2)) for ss in S])
  ret += p4**2*array([complex(psi(1,p3 + ss*p4)) for ss in S])
  ret += -p6**2*array([complex(psi(1,p5 + ss*p6)) for ss in S])
  ret += -p8**2*array([complex(psi(1,p7 + ss*p8)) for ss in S])
  ret += -p10**2*array([complex(psi(1,p9 + ss*p10)) for ss in S])
  return ret
logderivative2 = frompyfunc(fp,11,1)
