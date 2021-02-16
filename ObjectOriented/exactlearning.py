import numpy as np
from scipy.special import loggamma
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from math import frexp
from mpmath import mp, hyper, nstr, hyperu

##
def BFGS_search(p0):
  print("*** Initial Guess ***")
  res = minimize(self.real_diff,p0,method = 'BFGS', tol=1e-6)
  print("Params: ",res.x)
  print("Loss: ",res.fun)
  print("*** Refined Guess ***")
  popt=res.x
  print("Loss (including imaginary branches):",complex_diff(popt))
  res = minimize(complex_diff,popt,method = 'BFGS', tol=1e-8)
  print("Params: ",res.x)
  print("Loss: ",res.fun)
  popt=res.x
  fit = fingerprint(popt)
  print("Final Loss:",complex_diff(popt))
  return popt

mp.dps = 16; mp.pretty = True
np.seterr(divide = 'raise')
from scipy.stats import norm
def weighted_avg_and_std(values, weights):
  average = np.average(values, weights=weights)
  variance = np.average((values-average)**2, weights=weights)
  return (average, np.sqrt(variance))

twopi = 2*np.pi
twopi_rec = 1/twopi
pi_rec = 1/np.pi

## A function to get the result in the principle branch
def wr(x): return x - np.sign(x)*np.ceil(np.abs(twopi_rec*x)-0.5)*twopi
wrap = np.vectorize(wr)

## Vectorised difference function
def real_diff(p): #return np.sum(np.abs(func(p)-logmoments))
  A = fingerprint(p)
  B = np.abs(np.real(A)-real_logm)
  B = np.maximum(0.0,B-real_log_diff)
  return np.mean(B)

## Vectorised difference function
def complex_diff(p): #return np.sum(np.abs(func(p)-logmoments))
  A = fingerprint(p)
  B = np.abs(np.real(A)-real_logm)
  B = np.maximum(0.0,B-real_log_diff)
  C = np.abs(wrap(np.imag(A)-imag_logm))
  C = np.maximum(0.0,C-imag_log_diff)
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

## Set the tag here
tag = "Linear_0"

print("*** Assembling Dictionary ***")
## Load a dictionary of constants
#constants_dict = { i:j for i,j in zip(np.load("CONSTANTS_KEYS.npy"),np.load("CONSTANTS_VALUES.npy")) }

constants_dict = {}

## Fixes
constants_dict["0"]=0
constants_dict["Pi"] = np.pi
#constants_dict["e"] = np.exp(1)
#constants_dict["polylog(2,2/5)"] = 0.449282974471281664464733402376319384455327269535266637375904
#constants_dict["FresnelC(1)"]=0.77989340037682282947420641365
#constants_dict["sinh(1/4)"]= 0.252612316808168307914125150542
#constants_dict["tanh(1)"] = 0.7615941559557648881194582826047

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

### Define a rational/algebraic etc. solution space.

PT = np.array([-1,-2,-1/2,-3/4,3/4,3/2,5/2,2/3,np.sqrt(np.sqrt(2)),1/np.sqrt(np.sqrt(2)),1e-6,1,2,3,4,1/2,1/3,1/4,1/5,np.sqrt(2),np.sqrt(3),1/np.sqrt(2),1/np.sqrt(3),np.pi,1.0/np.pi])
#PT = np.reshape([PT,-PT],2*len(PT))

if(False):
  values = []
  index_to_drop = []
  A = len(constants_dict.keys())
  count = 0
  for i in constants_dict.keys():
    if(constants_dict[i] not in values): values.append(constants_dict[i])
    else: index_to_drop.append(i)
    count +=1
    print(count/A)
  
  for i in index_to_drop:
    del constants_dict[i]
  
  PT = np.array(list(constants_dict.values()))
  Pkeys = np.array(list(constants_dict.keys()))
  
  np.save("clean_values",PT)
  np.save("clean_keys",Pkeys)

PT = np.load("clean_values.npy")
PTkeys = np.load("clean_keys.npy")

reverse_dict = { i:j for i,j in zip(PT,PTkeys)}

PT[0]=1e-7

N_terms = 13
## Scaled
def fingerprint(p):
  ret = np.log(p[0]**2) ## A constant factor
  ret += s_values*np.log(p[1]**2) ## C^s for some C, together with previous cover most prefactors 
  ret += loggamma(p[2]+ p[3]*s_values) ## A flexible gamma 
  ret += loggamma(p[4] + p[5]*s_values) ## A flexible gamma 
  hyp = [complex(hyper([p[6]*s+p[7],p[8]+p[9]*s],[p[10]+p[11]*s],p[12])) for s in s_values] ## slow generalised_hypergeom
  ret += np.log(hyp)
  # s_values**2 * np.log(p[6]**2) #+ s**3 * np.log(p[3]**2) + s**4 * np.log(p[4]**2)  ## Strange series temrs
  #ret += np.log(1 + p[5]*s_values + p[6]*s_values**2 + p[7]*s_values**3 + p[8]*s_values**4) ## Log of polynomial
  return ret 

#p0 = np.ones(N_terms) + (0.5- np.random.rand(N_terms))

observations = []
losses = []


def categorical_solve(nits, L_in=None, P_in=None):
  C_size = len(PT)
  #static = np.array(range(N_terms))
  if(L_in == None): L = 0.001*np.ones((N_terms,C_size))
  C = 0.001*np.ones((N_terms,C_size)) 
  #K = np.random.choice(range(C_size),size=10,N_terms,replace=True)
  p = PT[K]
  l = complex_diff(p)
  Q = [[ np.exp(-np.abs(K[i]-PT[j]))/l for j in range(C_size)] for i in range(N_terms)]
  N = [[ np.exp(-np.abs(K[i]-PT[j])) for j in range(C_size)] for i in range(N_terms)]
  
  L += Q
  C += N
  
  ## Probability distribution over elements
  if(P_in == None):
    P = L/C
    N = np.sum(P,axis =1)
    P = P / N[:,None]
    N = np.sum(P,axis =1)

  #I.e. a n array of differences and sorted list...
  ## Add in an additional parameter choice which isn't in the list? (Some kind of solver?)
  ## Add in a routine that sets certain elements of P to zero after they drop below a threshold (number of observations)?
  
  losses = []
  for i in range(nits):
    power = 1 + i/1000
    K = np.array([np.random.choice(range(C_size),replace=True, p = pp) for pp in P])
    p = PT[K]
    try:
      l = complex_diff(p)
    except: 
      l = 100
    if(l>100): l = 100
    #l = 0.01+np.random.random()
    print(l)
    if(l<1e-6): return L, P
    Q = [[ np.exp(-np.abs(K[i]-PT[j]))/l for j in range(C_size)] for i in range(N_terms)]
    N = [[ np.exp(-np.abs(K[i]-PT[j])) for j in range(C_size)] for i in range(N_terms)]
    L += Q
    C += N
    P = L/C
    N = np.sum(P,axis =1)
    P = (P / N[:,None])**power
    N = np.sum(P,axis =1)
    P = (P / N[:,None])
    #if(i%100==0):
    #  i = np.transpose(np.argwhere(P<1e-3))
    #  L[i[0],i[1]] = 0
    #  P = L/C
    #  N = np.sum(P,axis =1)
    #  P = P / N[:,None] 
  return L, P


if(False):
  from scipy.stats import norm
  def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))
  
  
  ## First carry out a random set of experiments
  for i in range(1000):
    p0 = np.random.uniform(low=np.amin(PT),high=np.amax(PT),size=N_terms)
    score = complex_diff(p0)
    observations.append(p0)
    losses.append(score)
  
  for k in range(1):
    O = np.array(observations)
    L = np.array(losses)  
    ## Now for each parameter, derive a normal distribution 
    MS = np.array([ weighted_avg_and_std(np.transpose(O)[i], 1/L) for i in range(N_terms)])
    print(MS)
    for i in range(100):
      p0 = [ np.random.normal(loc=m,scale = 2*s) for m,s in MS]
      score = complex_diff(p0)
      observations.append(p0)
      losses.append(score)
  
  for k in range(100):
    O = np.array(observations)
    L = np.array(losses)  
    ## Now for each parameter, derive a normal distribution 
    MS = np.array([ weighted_avg_and_std(np.transpose(O)[i], 1/L) for i in range(N_terms)])
    print(MS)
  
    ## Consider the list of solutions weighted by the normals distributions
    PT_weights = [ [norm(loc=m,scale=s).pdf(k) for k in PT] for m,s in MS]
    PT_weights = [ a/np.sum(a) for a in PT_weights ]
    Ps = np.transpose(np.array([ np.random.choice(PT,size=10,p=p) for p in PT_weights ]))
    for p in Ps:
     score = complex_diff(p)
     observations.append(p)
     losses.append(score)
    print("Best Score:",np.amin(losses))
  
  print("Best Params:",observations[np.argmin(losses)])
  
  print(losses)
  print(observations)
  
## Currently broken
def plots(s,logm,fit): 
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

