import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

logmoments = np.load("logmoments_Circ_200.npy")
moments = np.load("moments_Circ_200.npy")
s_values = np.load("s_values_Circ_200.npy")
#real_error = np.load("real_error_Harmonic_4.npy")
#imag_error = np.load("imag_error_Harmonic_4.npy")


N_base = 7
N_constant = 0
N_plus = 0
N_minus = 0
PPT = 2
N_params_shift = N_base + N_constant + PPT*N_plus + PPT*N_minus

#[ 2.95097942e-01  1.18919127e+00 -4.48451244e-06  5.00006383e-01
#          2.64114837e+00 -3.52159930e+00  3.52146922e+00]

#[0.50135511 1.18919745 0.91495885 1.21996151]
# [0.45806836 1.09606426 1.46136098]
# [0.50460287 0.90322771 1.2042556 ]
# [0.91993398 1.22652979]
# 2.747983125106151e-06
# [0.91993397 1.22652979]

#[1.01604773e+00 1.18920257e+00 1.85735002e-06 5.00001790e-01                                                                                                                                                                                  2.22779696e-01 2.97040194e-01 2.97039154e-01] 


## Scaled
def func(p):
  A = loggamma(p[2] + p[3]*s_values) 
  B = np.log(p[0]**2) 
  C = s_values*np.log(p[1]**1) #+ s**2 * np.log(p[2]**2) + s**3 * np.log(p[3]**2) + s**4 * np.log(p[4]**2)
  polynom = np.log(p[4] - p[5]*s_values + p[6]*s_values**2)
  #constant = loggamma(p[2]) - loggamma(p[3])
  #off = N_base + N_constant 
  #plus = np.sum([ loggamma(p[off + PPT*k + 0]+ p[off + PPT*k + 1]*s) for k in range(N_plus)])
  #off = N_base + N_constant + PPT*N_plus
  #minus = np.sum([ -loggamma(p[off + PPT*k + 0]+p[off + PPT*k + 1]*s) for k in range(N_minus)])
  return A + B + C + polynom # + plus - minus

## Allow for nearby branches in the solution
def spc(m,sr,si,*p):
  qq = np.imag(func(sr,si,*p))
  ## Allow for 5 branches
  a = [(m - qq + k*2*np.pi)**2 for k in range(-2,3)]
  return np.amin(a)

## The difference to minimize
#def diff(p,S_R,S_I,M_R,M_I):
#  ## Add a regularisation term to force real inputs (s) to have real outputs (i.e. zero imaginary part) 
#  loss_real = np.sum([ (m - np.real(func(sr,si,*p)))**2 for sr,si,m in zip(S_R,S_I,M_R)])
#  loss_imag = np.sum([ spc(m,sr,si,*p) for sr,si,m in zip(S_R,S_I,M_I)])
#  ret = loss_real + loss_imag
#  print(p)
#  print(ret)
#  return ret


## Chop up
real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)
real_m = np.real(moments)
imag_m = np.imag(moments)

## Vectorised difference function
def diff(p): #return np.sum(np.abs(func(p)-logmoments))
  A = func(p)
  B = np.abs(np.real(A)-real_logm)
  C = np.abs(np.imag(A)-imag_logm)
  #TB = np.less(B,real_error)
  #TC = np.less(C,imag_error)
  #RB = np.where(TB,0.0,B)
  #RC = np.where(TC,0.0,C)
  return np.mean(B+C)


p0 = np.ones(N_params_shift) + 0.3*(0.5- np.random.rand(N_params_shift))
if(True):
  res = minimize(diff,p0,method = 'BFGS', tol=1e-8)
  print(res)
  popt=res.x

  fit = func(popt)
  print("Final Loss:",diff(popt))
 
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

