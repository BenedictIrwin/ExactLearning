import numpy as np
from scipy.special import loggamma, gammaln, gamma
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from mpl_toolkits import mplot3d
np.seterr(divide = 'raise')

with open("constants.txt") as f:
  flines = f.readlines()
  print(len(flines))
  strr = ""
  for i in flines: strr = strr+i
  flines = strr
  symbols = ['.\\','\\','\n']
  for i in symbols: flines = flines.replace(i,"")
  arr = eval(flines)
  arr = np.array(arr)
  constants = arr

with open("names_format.txt") as f:
  flines = f.readlines()[0]
  print(flines)
  arr = eval(flines)
  arr = [i.strip() for i in arr]
  print(arr)
  print(len(arr))
 
constants = np.concatenate(constants)
print(constants.shape)  

dic = {i:j for i,j in zip(arr,constants)}
print(dic)

np.save("CONSTANTS_KEYS",arr)
np.save("CONSTANTS_VALUES",constants)


exit()

with open("names.txt") as f:
  flines = f.readlines()
  flines = "".join(flines)
  flines = flines.split("\n")
  flines = "".join(flines)

  depth = 0
  print("[")
  for i in flines:
    if(i=="("): depth +=1
    if(i==")"): depth -=1
    if(i == "["):
      depth+=1
      if(depth<2): print('"',end="")
      else: print("[",end="")
      continue
    if( i == ']'):
      depth-=1
      if(depth<1): print('"',end="")
      else: print("]",end="")
      continue
    if( i == "," and depth == 1): 
      print('","',end="")
      continue
    print(i,end="")
  print("]")
  exit()

with open("names.txt") as f:
  flines = f.readlines()
  print(len(flines))
  strr = ""
  for i in flines: strr = strr+i
  flines = strr
  symbols = ['.\\','\\','\n','[']

  K = flines.split("\n")
  strr = ""
  for i in K: strr+=i
  strr = strr.replace("),","),\n").split("\n")
  for i in range(len(strr)):
    if("hypergeom" in strr[i]): strr[i] = strr[i].replace("[hyper","hyper") 
    else: strr[i] = strr[i].replace("[","") 
  for i in range(len(strr)):
    q = len(strr[i])
    if(strr[i][-1]==","): strr[i] = strr[i][0:q-1].strip()

  for i in strr: print(i)
  exit()

  for i in symbols: flines = flines.replace(i,"")
  print(flines)
  exit()
  
  
  flines = flines.split("]")
  arr = [ eval('"'+k.replace(",",'","')+'"') for k in flines]
  arr = [[j.strip() for j in i] for i in arr]
  names = arr

for i,j in zip(constants,names):
  print(i)
  print(j)
  for a,b in zip(i,j):
    print(a,b)

exit()

## Load a dictionary of constants
constants_dict = np.load("CONSTANTS_DICT.npy")

logmoments = np.load("logmoments_Harmonic_4.npy")
moments = np.load("moments_Harmonic_4.npy")
s_values = np.load("s_values_Harmonic_4.npy")
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
  C = s_values*np.log(p[1]**2) #+ s**2 * np.log(p[2]**2) + s**3 * np.log(p[3]**2) + s**4 * np.log(p[4]**2)
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

twopi = 2*np.pi
twopi_rec = 1/twopi
pi_rec = 1/np.pi

## A function to get the result in the principle branch
def wr(x): return x - np.sign(x)*np.ceil(np.abs(twopi_rec*x)-0.5)*twopi
wrap = np.vectorize(wr)

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

  ## Sqrt 2,3

  ## Pi



p0 = np.ones(N_params_shift) + (0.5- np.random.rand(N_params_shift))
if(True):
  print("*** Initial Guess ***")
  res = minimize(diff,p0,method = 'BFGS', tol=1e-8)
  print("Params: ",res.x)
  print("Loss: ",res.fun)
  print("*** Refined Guess ***")
  popt=res.x
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

