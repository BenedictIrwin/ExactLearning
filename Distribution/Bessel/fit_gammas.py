import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

logmoments = np.load("logmoments_bessel.npy")
s_values = np.load("s_values_bessel.npy")

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

N_base = 4
N_constant = 0
N_plus = 0
N_minus = 0
N_params_shift = N_base + N_constant + 2*N_plus + 2*N_minus

## Scaled
def func(sr,si, *p):
  s = sr+1j*si
  base =  s*np.log(p[0]**2) + np.log(p[1]**2) + p[3]*loggamma(p[2]*s) 
  #constant = loggamma(p[2]) - loggamma(p[3])
  off = N_base + N_constant
  plus = np.sum([ loggamma(p[off + 2*k]+ p[off + 2*k + 1]*s) for k in range(N_plus)])
  off = N_base + N_constant + 2*N_plus
  minus = np.sum([ loggamma(p[off + 2*k]+p[off + 2*k + 1]*s) for k in range(N_minus)])
  return base + plus - minus

## Allow for nearby branches in the solution
def spc(m,sr,si,*p):
  qq = np.imag(func(sr,si,*p))
  ## Allow for 5 branches
  a = [(m - qq + k*2*np.pi)**2 for k in range(-2,3)]
  return np.amin(a)

## The difference to minimize
def diff(p,real_s,imag_s,real_logm,imag_logm):
  print(p)
  #reg1 =  500*( np.real(func(1,0,*p)) )**2 + 500*( np.imag(func(1,0,*p)))**2
  #reg2 = 500*( np.real(func(2,0,*p)) +0.88118109562192976936243092633063780660300419688979154688)**2
  
  #reg_s = [q for q in range(1,10)]
  #reg3 = 500*np.sum([ np.imag(func(q,0,*p))**2 for q in reg_s]) ## Keeping the output real for these integer moments

  ## Add a regularisation term to force real inputs (s) to have real outputs (i.e. zero imaginary part) 
  loss_real = np.sum([ (m - np.real(func(sr,si,*p)))**2 for sr,si,m in zip(real_s,imag_s,real_logm)])
  loss_imag = np.sum([ spc(m,sr,si,*p) for sr,si,m in zip(real_s,imag_s,imag_logm)])
  #loss_params = 3000*np.sum([pp**2 for pp in p])
  ret = loss_real + loss_imag
  print(ret)
  return ret

p0 = np.random.rand(N_params_shift)
p0 = np.ones(N_params_shift) + 0.2 * np.random.rand(N_params_shift)

## Ideal result
#p0 = [np.sqrt(2),0.5,0.5,2.0]

real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)




if(True):
  res = minimize(diff,p0,args = (real_s,imag_s,real_logm,imag_logm),method = 'BFGS')
  print(res)
  popt=res.x

  fit = np.array([ func(sr,si,*popt) for sr,si in zip(real_s,imag_s)])
  loss_real = np.sum([ (m - np.real(func(sr,si,*popt)))**2 for sr,si,m in zip(real_s,imag_s,real_logm)])
  loss_imag = np.sum([ spc(m,sr,si,*popt) for sr,si,m in zip(real_s,imag_s,imag_logm)])
  print("Final Loss:",loss_real+loss_imag)
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, real_logm, c=real_logm, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.real(fit), c=np.real(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$Re(\log E[x^{s-1}])$')
  plt.legend()
  plt.show()
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, imag_logm, c=imag_logm, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.imag(fit), c=np.imag(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$Im(\log E[x^{s-1}])$')
  plt.legend()
  plt.show()

  p_best = popt

