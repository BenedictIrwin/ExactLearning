import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from mpl_toolkits import mplot3d
from math import frexp
np.seterr(divide = 'raise')

print("*** Assembling Dictionary ***")
## Load a dictionary of constants
#constants_dict = { i:j for i,j in zip(np.load("CONSTANTS_KEYS.npy"),np.load("CONSTANTS_VALUES.npy")) }

constants_dict = {}

## Fixes
constants_dict["0"]=0
constants_dict["Pi"] = np.pi
#constants_dict["e"] = np.exp(1)
#constants_dict["polylog(2,2/5)"] = 0.449282974471281664464733402376319384455327269535266637375904
#constants_dict["FresnelC(1)"]=0.77989340037682282947420641365269013663062570813632096010313358317807176097910889010877870730527852815898902782
#constants_dict["sinh(1/4)"]= 0.25261231680816830791412515054205790551975428742766080748809496530198107690685937906065370209611987414876924736
#constants_dict["tanh(1)"] = 0.76159415595576488811945828260479359041276859725793655159681050012195324457663848345894752167367671442190275970

for i in range(2,7): constants_dict["{}".format(i)] = i

keys_to_drop = []
for i in constants_dict.keys():
  v1 = constants_dict[i]
  if(np.abs(v1)>1e20): keys_to_drop.append(i)

print(keys_to_drop)
for key in keys_to_drop: del constants_dict[key]

## By repeating we get a much richer dictionary
for repeats in range(2):
  new_keys = []
  new_values = []
  ## Looping over entries
  for i in constants_dict.keys():
    v1 = constants_dict[i]
    if(v1!=0):
      new_keys.append("recip({})".format(i))
      new_values.append(1/v1)
    new_keys.append("minus({})".format(i))
    new_values.append(-v1)
    if(v1>0):
      new_keys.append("sqrt({})".format(i))
      new_values.append(np.sqrt(v1))
      new_keys.append("3rt({})".format(i))
      new_values.append(v1**(1/3))
      new_keys.append("4rt({})".format(i))
      new_values.append(v1**(1/4))
      new_keys.append("log({})".format(i))
      new_values.append(np.log(v1))
    for j in constants_dict.keys():
      v2 = constants_dict[j]
      new_keys.append("sum({},{})".format(i,j))
      new_values.append(v1 + v2)
      new_keys.append("diff({},{})".format(i,j))
      new_values.append(v1 - v2)
      new_keys.append("prod({},{})".format(i,j))
      new_values.append(v1 * v2)
      if(v2!=0):
        new_keys.append("quot({},{})".format(i,j))
        new_values.append(v1 / v2)
  
  constants_dict.update(zip(new_keys,new_values))


print("*** Loading Data ***")

logmoments = np.load("logmoments_EXPWELL_0.npy")
moments = np.load("moments_EXPWELL_0.npy")
s_values = np.load("s_values_EXPWELL_0.npy")


N_base = 7
N_constant = 0
N_plus = 0
N_minus = 0
PPT = 2
N_params_shift = N_base + N_constant + PPT*N_plus + PPT*N_minus

## A function to get the result in the principle branch
def wr(x): return x - np.sign(x)*np.ceil(np.abs(twopi_rec*x)-0.5)*twopi
wrap = np.vectorize(wr)

## Scaled
def func(p):
  ret = p[4]*loggamma(p[1] + p[2]*s_values)
  ret += np.log(p[0]**2) 
  ret += s_values*np.log(p[3]**2) # s_values**2 * np.log(p[6]**2) #+ s**3 * np.log(p[3]**2) + s**4 * np.log(p[4]**2)
  ret += np.log(1 + p[5]*s_values + p[6]*s_values**2)
  #constant = loggamma(p[2]) - loggamma(p[3])
  #off = N_base + N_constant 
  #plus = np.sum([ loggamma(p[off + PPT*k + 0]+ p[off + PPT*k + 1]*s) for k in range(N_plus)])
  #off = N_base + N_constant + PPT*N_plus
  #minus = np.sum([ -loggamma(p[off + PPT*k + 0]+p[off + PPT*k + 1]*s) for k in range(N_minus)])
  return np.real(ret) + 1j*wrap(np.imag(ret))

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

twopi = 2*np.pi
twopi_rec = 1/twopi
pi_rec = 1/np.pi


## Vectorised difference function
def sniff(p): #return np.sum(np.abs(func(p)-logmoments))
  A = func(p)
  B = np.abs(np.real(A)-real_logm)
  C = np.abs(wrap(np.imag(A)-imag_logm))
  return np.mean(B+C)

def analyse(p):
  print("Situation Report")
  num = len(p)
  ## Check for zero elements
  for i in range(num): 
    if(abs(p[i])<1e-4): print("p[{}] ~ 0, consider setting to zero!".format(i))
  ## Check for elements which are of similar magnitude
  for i in range(num):
    for j in range(i):
      if(abs(abs(p[i])-abs(p[j])) < 1e-4):
        if(np.sign(p[i])==np.sign(p[j])): print("p[{}] ~ p[{}], consider reducing to one parameter!".format(i,j))
        else: print("p[{}] ~ -p[{}], consider reducing to one parameter!".format(i,j))
  ## Special constants
  
  ## Integers
  for i in range(num):
    for k in range(1,10):
      if(abs(p[i]-k)<1e-3): print("p[{}] ~ {}".format(i,k))
      if(abs(p[i]**2-k)<1e-3): print("p[{}]^2 ~ {}".format(i,k))
      if(abs(p[i]**3-k)<1e-3): print("p[{}]^3 ~ {}".format(i,k))
      if(abs(p[i]**4-k)<1e-3): print("p[{}]^4 ~ {}".format(i,k))
      if(abs(1/p[i]-k)<1e-3): print("p[{}] ~ 1/{}".format(i,k))
      if(abs(1/p[i]**2-k)<1e-3): print("p[{}]^2 ~ 1/{}".format(i,k))
      if(abs(1/p[i]**3-k)<1e-3): print("p[{}]^3 ~ 1/{}".format(i,k))
      if(abs(1/p[i]**4-k)<1e-3): print("p[{}]^4 ~ 1/{}".format(i,k))

  ## Rational Approximation
  #for i in range(num): 

  c_vals = np.array(list(constants_dict.values()))
  c_keys = np.array(list(constants_dict.keys())) 

  ## Constants
  for i in range(num):
    diff = np.abs(p[i]-c_vals)
    index = np.argmin(diff)
    lowest = np.amin(diff)
    if(lowest < 1e-3):
      name = c_keys[index]
      value = c_vals[index]
      print("p[{}] ~ {} : i.e. {} ~ {}".format(i,name,p[i],value))


p0 = np.ones(N_params_shift) + (0.5- np.random.rand(N_params_shift))

if(True):
  print("*** Initial Guess ***")
  res = minimize(diff,p0,method = 'BFGS', tol=1e-8)
  print("Params: ",res.x)
  print("Loss: ",res.fun)
  print("*** Refined Guess ***")
  popt=res.x
  print("Refined Loss:",sniff(popt))
  res = minimize(sniff,popt,method = 'BFGS', tol=1e-8)
  print("Params: ",res.x)
  print("Loss: ",res.fun)
  popt=res.x
  fit = func(popt)
  print("Final Loss:",sniff(popt))

  analyse(popt)

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

