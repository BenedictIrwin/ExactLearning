from AdvancedFunctions import *
from numpy import array, frompyfunc
def fp(p0,p1):
  S = array([(1.78882126703924-2.975078104256949j), (2.770531531407552-0.7665863253971379j), (2.1534471216504185-0.9767657705952901j), (6.377266866935955+3.0018429911336506j), (5.682491548732583-1.6773522950490747j)])
  ret = 0
  ret += p1**2*np.trigamma(p0 + S*p1)
  return ret
logderivative2 = frompyfunc(fp,2,1)
