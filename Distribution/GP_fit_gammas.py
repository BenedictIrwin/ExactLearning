import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from bayes_opt import BayesianOptimization


np.seterr(divide = 'raise')

logmoments = np.load("logmoments_roots_simp.npy")
moments = np.load("moments_roots_simp.npy")
s_values = np.load("s_values_roots_simp.npy")

N_base = 3
N_constant = 0
N_plus = 1
N_minus = 1
N_params_shift = N_base + N_constant + 3*N_plus + 3*N_minus

## Scaled
def func(sr,si, *p):
  s = sr+1j*si
  base =  p[0]*s*np.log(p[1]**2) + np.log(p[2]**2)
  #constant = loggamma(p[2]) - loggamma(p[3])
  off = N_base + N_constant 
  PPT = 3
  plus = np.sum([ p[off + PPT*k]*loggamma(p[off + PPT*k + 1]+ p[off + PPT*k + 2]*s) for k in range(N_plus)])
  off = N_base + N_constant + 3*N_plus
  minus = np.sum([ p[off + PPT*k]*loggamma(p[off + PPT*k + 1]+p[off + PPT*k + 2]*s) for k in range(N_minus)])
  return np.exp(base + plus - minus)


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
  return base + plus - minus

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


def real_square_diff(m,sr,si,*p): return np.square(m - np.real(func(sr,si,*p)))


## The difference to minimize
def diff(p):   
  ## Add a regularisation term to force real inputs (s) to have real outputs (i.e. zero imaginary part) 
  loss_real = np.sum([ real_square_diff(m,sr,si,*p)  for sr,si,m in zip(real_s,imag_s,real_m)])
  loss_imag = np.sum([ spc(m,sr,si,*p) for sr,si,m in zip(real_s,imag_s,imag_m)])
  ret = loss_real + loss_imag
  return -ret


from bayes_opt import UtilityFunction
utility = UtilityFunction(kind="ei", kappa = 1, xi=0.001)

## Generate variables p_0 through p_{N-1}
bounds_dict = { 'p{}'.format(k) : (-3,3) for k in range(N_params_shift) }
# Bounded region of parameter space
p_strings = ["p{}".format(k) for k in range(N_params_shift)]


for round in range(10):
  optimizer = BayesianOptimization(f=None, pbounds=bounds_dict,random_state=np.random.randint(1))
  optimizer.set_gp_params(normalize_y=True)
  for i in range(50):
    next_point = optimizer.suggest(utility)
    params = [ next_point[k] for k in p_strings]
    target = vec_diff(params)
    print(target)
    if(np.isfinite(target) and abs(target) < 1e40): optimizer.register(params=next_point, target=target)
  bound = np.quantile([a["target"] for a in optimizer.res],0.3)
  good_results = [ res for res in optimizer.res if res["target"]<bound ]
  ## Here get the new bounds and restart? 
  
  print(good_results)
  bounds_dict = {}
  for st in p_strings:
    temp = [res["params"][st] for res in good_results]
    bound = (np.amin(temp),np.amax(temp))
    bounds_dict[st]=bound
  
  print(optimizer.max)
print(optimizer.max)


exit()


if(True):
  #res = minimize(diff,p0,args = (real_s,imag_s,real_m,imag_m),method = 'BFGS')
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

