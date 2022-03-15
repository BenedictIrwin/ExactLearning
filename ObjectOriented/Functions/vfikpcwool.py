from numpy import log
from scipy.special import loggamma
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([array([2.5418735 +1.92342812j, 2.34927255-0.68442452j,
       1.90617061+2.5786939j , 4.40539737+2.74591471j,
       1.02204305-0.67240178j, 2.0795091 +2.21555441j,
       1.0574252 +2.59788721j, 2.14109946+0.33843038j,
       4.68101896-0.18651467j, 2.8058358 -0.83015393j])])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
