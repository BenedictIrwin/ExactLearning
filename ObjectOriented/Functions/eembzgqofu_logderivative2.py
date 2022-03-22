
from mpmath import psi
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([3.49619818-1.4791873j , 4.75913576-0.35484671j,
       2.33125916+0.55841508j, 2.26844643+1.11699897j,
       2.54223396-0.15659784j, 2.50661832+0.53853753j,
       4.21808384+1.64796235j, 1.55214089+1.13360313j,
       3.04024421-1.3119992j , 4.68555186+1.04888199j])
  ret = 0
  ret += 0
  ret += 0
  ret += p3**2*array([complex(psi(1,p2 + ss*p3)) for ss in S])
  return ret
logderivative2 = frompyfunc(fp,4,1)


print(psi(1, 2 + 3))

pp = [1,2,3,4]
res = logderivative2(*pp)
print(res)
