from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1):
  S = array([(2.517099387901605+1.3926710109956923j), (2.5735281119019104-0.6342857576714316j), (3.688420690919453+1.961722126203271j), (5.963612592944012-1.0528403377092532j), (6.143616827886945-0.9643253374491723j)])
  ret = 0
  ret += log(p0**2)
  ret += -log(1.0 + S*p1)
  return ret
fingerprint = frompyfunc(fp,2,1)
