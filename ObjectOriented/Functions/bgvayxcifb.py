from scipy.special import loggamma
from numpy import log
from numpy import array, frompyfunc
def fp(p0,p1,p2,p3,p4,p5,p6):
  S = array([(6.006858980447979+1.5076971158555654j), (4.594994119083825-0.8674380358008698j), (4.541237259642635+1.5546601379267058j), (2.4779443262585623-2.209476569753647j), (1.4834005033700217+2.829377522687338j), (2.336128589056246-2.381327023150575j), (6.941052900696043+2.819179807855809j), (5.7310274904640375+0.2895336715014265j), (5.957598025550404+0.6410200526608674j), (5.852288405058515-2.1676707443521748j)])
  ret = 0
  ret = np.add(ret,log(p0**2),out=ret,casting='unsafe')
  ret = np.add(ret,S*log(p1**2),out=ret,casting='unsafe')
  ret = np.add(ret,loggamma(p3*p2 + S*p3),out=ret,casting='unsafe')
  ret = np.add(ret,-loggamma(p5*p4 + S*p5),out=ret,casting='unsafe')
  ret = np.add(ret,-log(1.0 + S*p6),out=ret,casting='unsafe')
  return ret
fingerprint = frompyfunc(fp,7,1)
