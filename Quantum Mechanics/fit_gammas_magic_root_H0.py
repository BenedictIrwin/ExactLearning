import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

logmoments = np.load("logmoments_Harmonic.npy")
moments = np.load("moments_Harmonic.npy")
s_values = np.load("s_values_Harmonic.npy")

N_base = 7
N_constant = 0
N_plus = 0
N_minus = 0
PPT = 2
N_params_shift = N_base + N_constant + PPT*N_plus + PPT*N_minus

## Scaled
def func(sr,si, *q):
  p=list(q)
  s = sr+1j*si
  base =  loggamma(p[3] + p[2]*s) + np.log(p[0]**2) + s*np.log(p[1]**2) #+ s**2 * np.log(p[2]**2) + s**3 * np.log(p[3]**2) + s**4 * np.log(p[4]**2)
  polynom = np.log(p[4] + p[5]*s + p[6]*s**2)
  #constant = loggamma(p[2]) - loggamma(p[3])
  off = N_base + N_constant 
  plus = np.sum([ loggamma(p[off + PPT*k + 0]+ p[off + PPT*k + 1]*s) for k in range(N_plus)])
  off = N_base + N_constant + PPT*N_plus
  minus = np.sum([ -loggamma(p[off + PPT*k + 0]+p[off + PPT*k + 1]*s) for k in range(N_minus)])
  return base + polynom + plus - minus

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

#p0 = [0.001, 2.0, np.sqrt(np.sqrt(np.pi)/4), 3.0, 0.0, 0.5, -1.0, 0.5, 0.5]

## Chop up
real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)
real_m = np.real(moments)
imag_m = np.imag(moments)

#p0 = np.array([ 2.68893128,  1.24850336, -0.4784735 ,  1.06710095,  1.09076523,
#            0.11482134,  1.20597782,  0.07102955,  0.33651599,  0.66499011,
#                    3.55413955,  1.04199151, -0.68654022,  3.09616407, -0.20412917])

## The difference to minimize
def diff(p):
  ## Add a regularisation term to force real inputs (s) to have real outputs (i.e. zero imaginary part) 
  loss_real = np.sum([ (m - np.real(func(sr,si,*p)))**2 for sr,si,m in zip(real_s,imag_s,real_logm)])
  loss_imag = np.sum([ spc(m,sr,si,*p) for sr,si,m in zip(real_s,imag_s,imag_logm)])
  ret = loss_real + loss_imag
  print(p)
  print(ret)
  return ret

### For each known value of s (complex)
### Solve for parameters

def value_func(p,a,b,c,d): return [(mr - np.real(func(sr,si,*p)))**2 for sr,si,mr in zip(a,b,c,d)]
#def value_func(p,a,b,c,d): return [(mr - np.real(func(sr,si,*p)))**2 + spc(mi,sr,si,*p) for sr,si,mr,mi in zip(a,b,c,d)]

res = root(value_func,p0,args=(real_s[0:7],imag_s[0:7],real_logm[0:7]), method = 'df-sane')
print(res)
exit()

if(True):
  res = minimize(diff,p0,args = (real_s,imag_s,real_logm,imag_logm),method = 'BFGS')
  print(res)
  popt=res.x

  fit = np.array([ func(sr,si,*popt) for sr,si in zip(real_s,imag_s)])
  loss_real = np.sum([ (m - np.real(func(sr,si,*popt)))**2 for sr,si,m in zip(real_s,imag_s,real_m)])
  loss_imag = np.sum([ spc(m,sr,si,*popt) for sr,si,m in zip(real_s,imag_s,imag_m)])
  print("Final Loss:",loss_real+loss_imag)
 
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
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, real_logm, c=real_logm, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.real(fit), c=np.real(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$\log Re(E[x^{s-1}])$')
  plt.legend()
  plt.show()
  
  ax = plt.axes(projection='3d')
  # Data for three-dimensional scattered points
  ax.scatter3D(real_s, imag_s, imag_logm, c=imag_logm, cmap='Reds', label = "Numeric")
  ax.scatter3D(real_s, imag_s, np.imag(fit), c=np.imag(fit), cmap='Greens', label = "Theoretical")
  ax.set_xlabel('Re(s)')
  ax.set_ylabel('Im(s)')
  ax.set_zlabel('$\log Im(E[x^{s-1}])$')
  plt.legend()
  plt.show()

  p_best = popt

