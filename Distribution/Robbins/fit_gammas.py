import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

logmoments = np.load("logmoments_Robbins.npy")
moments = np.load("moments_Robbins.npy")
s_values = np.load("s_values_Robbins.npy")

## Current best
#def func(s, a, b, c, d, e, f): 
#  return s*np.log(e) + np.log(f) + loggamma(s) - loggamma(a+s) + loggamma(b+s) - loggamma(c+s)

N_base = 4
N_plus = 5
N_minus = 5
N_params_shift = N_base + N_plus + N_minus

def func(s, *p):
  base =  s*np.log(p[0]) + np.log(p[1]) + loggamma(s) + loggamma(p[3]) - loggamma(p[4])
  plus = np.sum([ loggamma(p[N_base + k]+s) - loggamma(p[N_base + k]) for k in range(N_plus)])
  minus = np.sum([ loggamma(p[N_base + N_plus + k]+s) - loggamma(p[N_base + N_plus + k]) for k in range(N_minus)])
  return base + plus - minus

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
  print(p)
  print(ret)
  return ret

p0 = np.random.rand(N_params_shift)
p0 = np.ones(N_params_shift) + 0.2 * np.random.rand(N_params_shift)

## Chop up
real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)
real_m = np.real(moments)
imag_m = np.imag(moments)

p0 = array([-1.30082218e+00,  1.50129517e+00, -1.90581963e-02, -7.38286199e-01,
            4.20439949e+00,  1.60828280e+00,  1.60459098e+00,  8.62136038e-01,
                    1.15975519e+00, -6.09028514e-01,  4.55673055e+00,  1.03539051e-03,
                            9.54978847e-01,  3.35185776e+00,  1.14912016e+00])

if(True):
  res = minimize(diff,p0,args = (real_s,imag_s,real_m,imag_m),method = 'BFGS',tol=1e-2)
  print(res)
  popt=res.x

  fit = np.array([ func(sr,si,*popt) for sr,si in zip(real_s,imag_s)])
  loss_real = np.sum([ (m - np.real(func(sr,si,*popt)))**2 for sr,si,m in zip(real_s,imag_s,real_m)])
  loss_imag = np.sum([ spc(m,sr,si,*popt) for sr,si,m in zip(real_s,imag_s,imag_m)])
  print("Final Loss:",loss_real+loss_imag)
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, real_m, c=real_m, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.real(fit), c=np.real(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$Re(E[x^{s-1}])$')
  plt.legend()
  plt.show()
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, imag_m, c=imag_m, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.imag(fit), c=np.imag(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$Im(E[x^{s-1}])$')
  plt.legend()
  plt.show()

  p_best = popt

