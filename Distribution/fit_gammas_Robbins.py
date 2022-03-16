import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

S_samples = []
Q_samples = []

for i in range(3):
  N = 20000
  x_1 = np.random.random(N)
  y_1 = np.random.random(N)
  z_1 = np.random.random(N)
  x_2 = np.random.random(N)
  y_2 = np.random.random(N)
  z_2 = np.random.random(N)
  d = np.sqrt( (x_1-x_2)**2 + (y_1-y_2)**2 + (z_1 - z_2)**2 )
  
  ## Generate complex moments?
  s1 = np.random.uniform(low = 1, high =3, size = 1000)
  s2 = np.random.uniform(low = -np.pi, high =np.pi, size = 1000)
  s = [ t1 + t2*1j for t1,t2 in zip(s1,s2) ]
  q = [np.mean(np.power(d,ss-1)) for ss in s]
  S_samples.append(np.array(s))
  Q_samples.append(np.array(q))


### The model
N_base = 3
N_constant = 0
N_plus = 2
N_minus = 2
N_params_shift = N_base + N_constant + 3*N_plus + 3*N_minus

## Scaled
def func(sr,si, *p):
  s = sr+1j*si
  base =  s*np.log(p[0]**2) + np.log(p[1]**2) + p[2]*loggamma(s) 
  #constant = loggamma(p[2]) - loggamma(p[3])
  off = N_base + N_constant
  plus = np.sum([ p[off + 2*k]*loggamma(p[off + 2*k + 1]+ p[off + 2*k + 2]*s) for k in range(N_plus)])
  off = N_base + N_constant + 3*N_plus
  minus = np.sum([ p[off + 2*k]*loggamma(p[off + 2*k + 1]+p[off + 2*k + 2]*s) for k in range(N_minus)])
  return np.exp(base + plus - minus)

## Allow for nearby branches in the solution
def spc(m,sr,si,*p):
  qq = np.imag(func(sr,si,*p))
  ## Allow for 5 branches
  a = [(m - qq + k*2*np.pi)**2 for k in range(-2,3)]
  return np.amin(a)

## The difference to minimize
def diff(p,S_R,S_I,M_R,M_I):
  ## Add a regularisation term to force real inputs (s) to have real outputs (i.e. zero imaginary part) 
  loss_real = np.sum([ (m - np.real(func(sr,si,*p)))**2 for sr,si,m in zip(S_R,S_I,M_R)])
  loss_imag = np.sum([ spc(m,sr,si,*p) for sr,si,m in zip(S_R,S_I,M_I)])
  ret = loss_real + loss_imag
  #print(p)
  #print(ret)
  return ret

#p0 = np.random.rand(N_params_shift)
#p0 = np.ones(N_params_shift) + 0.2 * np.random.rand(N_params_shift)

p0 = np.array([-1.30082218e+00,  1.50129517e+00, -1.90581963e-02, -7.38286199e-01,
            4.20439949e+00,  1.60828280e+00,  1.60459098e+00,  8.62136038e-01,
                    1.15975519e+00, -6.09028514e-01,  4.55673055e+00,  1.03539051e-03,
                            9.54978847e-01,  3.35185776e+00,  1.14912016e+00])


pdiffs = [ np.abs(p0)*(np.random.rand(len(p0))-0.5) for kk in range(50) ]
all_diffs = []
## Chop up
for s,q in zip(S_samples,Q_samples):
  real_s = np.real(s)
  imag_s = np.imag(s)
  real_m = np.real(q)
  imag_m = np.imag(q)
  deltas = [1.0/diff(p0+dd,real_s,imag_s,real_m,imag_m) for dd in pdiffs]
  DD = deltas/np.sum(deltas)
  all_diffs.append(DD)


all_diffs = np.array(all_diffs)
all_diffs = np.sum(all_diffs, axis = 0 )

best_diff = np.einsum("i,ij->j",all_diffs,pdiffs)/np.sum(all_diffs)
print(p0+best_diff)
exit()

median = np.median(all_diffs)
print(median)

GOOD = []
BAD = []

## A strategy to estimate parameters
for i in range(len(all_diffs)):
  if(all_diffs[i] > median): GOOD.append(p0+pdiffs[i])
  else: BAD.append(p0+pdiffs[i])

GOOD = np.transpose(np.array(GOOD))
BAD = np.transpose(np.array(BAD))

print(GOOD)
print(BAD)

for g,b in zip(GOOD,BAD):
  plt.hist(g,bins=20,label="Good")
  plt.hist(b,bins=20,label="Bad")
  plt.legend()
  plt.show()


