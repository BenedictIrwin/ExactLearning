import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

logmoments = np.load("logmoments_NN1.npy")
moments = np.load("moments_NN1.npy")
s_values = np.load("s_values_NN1.npy")

N_base = 3
N_constant = 0
N_plus = 1
N_minus = 1
PPT = 2
N_params_shift = N_base + N_constant + PPT*N_plus + PPT*N_minus

## Scaled
def func(sr,si, *p):
  s = sr+1j*si
  base =  p[0]*s*np.log(p[1]**2) + np.log(p[2]**2)
  #constant = loggamma(p[2]) - loggamma(p[3])
  off = N_base + N_constant 
  plus = np.sum([ loggamma(p[off + PPT*k + 0]+ p[off + PPT*k + 1]*s) for k in range(N_plus)])
  off = N_base + N_constant + PPT*N_plus
  minus = np.sum([ -loggamma(p[off + PPT*k + 0]+p[off + PPT*k + 1]*s) for k in range(N_minus)])
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
  return ret

## Chop up
real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)
real_m = np.real(moments)
imag_m = np.imag(moments)

p0 = np.random.rand(N_params_shift)

reps = 200
params = []
losses = []
for rep in range(reps):
  p0 = np.ones(N_params_shift) + 0.2 * np.random.rand(N_params_shift)
  if(True):
    res = minimize(diff,p0,args = (real_s,imag_s,real_m,imag_m),method = 'BFGS')
    popt=res.x
  
    fit = np.array([ func(sr,si,*popt) for sr,si in zip(real_s,imag_s)])
    loss_real = np.sum([ (m - np.real(func(sr,si,*popt)))**2 for sr,si,m in zip(real_s,imag_s,real_m)])
    loss_imag = np.sum([ spc(m,sr,si,*popt) for sr,si,m in zip(real_s,imag_s,imag_m)])
    print("Final Loss:",loss_real+loss_imag)
    print(popt)
    params.append(popt)
    losses.append(loss_real+loss_imag)

print(np.array(params))
print(np.array(losses))

np.save("KNN_params",np.array(params))
np.save("KNN_losses",np.array(losses))















## Plotting
if(False):
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

