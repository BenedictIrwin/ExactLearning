from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3):
  S = array([(1.5869155565980124-3.0856856341159875j), (2.4840728681710966+0.9745469110552465j), (1.275052542845331+3.1011316666598274j), (2.381197204034405+0.7950343825174278j), (3.034902166277082-3.136355254903044j), (4.729374618267194+2.1238863511003574j), (4.6621535611549465-0.020844044781858084j), (1.4218526647897227+2.969388376760249j), (3.638920024099893-0.31699404318188806j), (4.745381810902532-0.36636251169684453j)])
  ret = 0
  ret += log(p0**2)
  ret += S*log(p1**2)
  ret += loggamma(p2 + S*p3)
  return ret
fingerprint = frompyfunc(fp,4,1)
