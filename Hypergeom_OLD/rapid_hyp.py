import numpy as np
from scipy.special import loggamma
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from math import frexp
from mpmath import mp, hyper, nstr, hyperu
from exactlearning import BFGS_search, analyse

mp.dps = 16; mp.pretty = True
np.seterr(divide = 'raise')

#twopi = 2*np.pi
#twopi_rec = 1/twopi
#pi_rec = 1/np.pi

## Set the tag here
tag = "Linear_0"

print("*** Loading Data ***")
N_d = 10
logmoments = np.load("logmoments_{}.npy".format(tag))[:N_d]
moments = np.load("moments_{}.npy".format(tag))[:N_d]
s_values = np.load("s_values_{}.npy".format(tag))[:N_d]
real_error = np.load("real_error_{}.npy".format(tag))[:N_d]
imag_error = np.load("imag_error_{}.npy".format(tag))[:N_d]

## Chop up
real_s = np.real(s_values)
imag_s = np.imag(s_values)
real_logm = np.real(logmoments)
imag_logm = np.imag(logmoments)
real_m = np.real(moments)
imag_m = np.imag(moments)

real_log_upper = np.real(np.log(moments + real_error))
real_log_lower = np.real(np.log(moments - real_error))
imag_log_upper = np.imag(np.log(moments + imag_error))
imag_log_lower = np.imag(np.log(moments - imag_error))

## The bounds to work with
real_log_diff = real_log_upper - real_log_lower
imag_log_diff = imag_log_upper - imag_log_lower



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

#from scipy.spatial import KDTree
### Define a point cloud
#points = [[[[[a,b,c,d] for d in PT] for c in PT] for b in PT] for a in PT]
#points = np.array(points)
#points = np.reshape(points,(len(PT)**4,4))
#points_tree = KDTree(points)

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
  L, P = categorical_solve(1000)
  
  for i in range(N_terms):
    q = np.quantile(P[i],0.75)
    m = np.argmax(P[i])
    indices = np.where(P[i] > q)
    terms = PT[indices]
    print("p[{}] ~ ".format(i),terms)
    print("Hypothesis: p[{}] ~ {}".format(i,PT[m]))
  
  for i in range(len(PT)):
    print(i,PT[i])
  
  for i in P:
    plt.bar(range(len(i)),i)
    plt.show()
  
  exit()

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
  
  
  #p_test = [1/np.sqrt(2),0.5,0.5,0.25]
  
  ## Loop here
  #res = points_tree.query(p_test,k=20)
  #new_indices = res[1]
  #vecs = points[new_indices]
  #scores = [complex_diff(p) for p in vecs]
  #print(scores)

p0= np.random.random(N_terms)

#p0 = [ 0.51238944,0.97451466,-0.01,0.4491124,0.12458327,0.82568312,0.20801154,0.27429931,0.73933532,0.16679021,0.5342653,0.90349894,0.31334464,  0.68688119]
p0 = [ 0.51238944,0.97451466,0.4491124,0.12458327,0.82568312,0.20801154,0.27429931,0.73933532,0.16679021,0.5342653,0.90349894,0.31334464,  0.68688119]

if(True):
  if(True):
    popt = BFGS_search(p0)
  else:
    print("BFGS Disabled")
    popt = p0
  print("** Searching for Algebraic Identity ***")
  ## Rational searcher
  
  ## Loop here
  #res = points_tree.query(popt,k=10)
  #new_indices = res[1]
  #vecs = points[new_indices]
  #scores = [complex_diff(p) for p in vecs]
  #best_score = np.argmin(scores)
  #print(vecs[best_score],scores[best_score])
   
  analyse(popt)

  ## Add these "best" solutions to the mix
  ## This gives us a chance at partially optimising the original solution
  PT = np.concatenate((PT,popt))
  PTkeys = np.concatenate((PTkeys,["BFGS_param_{}".format(i) for i in range(len(popt))]))
  reverse_dict = { i:j for i,j in zip(PT,PTkeys)}
  ##
  ## IMPORTANT IDEA
  ## CONSIDER FIRST ITERATING EACH PARAMETER IN TERMS OF NEARBY SOLUTIOSN WHILE KEEPING THE OTHER TERMS CONSTANT
  ## IF WE GET ANY HITS THIS IS PROMISING

  PT2 = [[k] for k in PT]
  value_tree = KDTree(PT2)
 
  CHOICES = []
  for i in range(len(popt)):
    k_query = 5
    nearest_k = value_tree.query([popt[i]],k=k_query)
    ## Get all the values which are within 0.1
    dists = nearest_k[0]
    inds = np.argwhere(dists <= 0.1)
    elements = nearest_k[1][inds]
    choice = [k[0] for k in PT[elements]]
    CHOICES.append(choice)
    print("p[{}] choose from {}".format(i,choice))
    
  ## Set up a score system for the choices
  P = np.zeros((len(popt),k_query))
  for i in range(len(CHOICES)):
    for j in range(len(CHOICES[i])):
      P[i,j]+=1
  N = np.sum(P,axis =1)
  P = (P / N[:,None])

  ## A probabilistic scoring approach
  if(False):
    ## Assemble all parameter combinations
    nits = 10*np.prod([len(ii) for ii in CHOICES])
    print("N iterations = {}".format(nits))
    ## Or run the weighted combinations analysis
 
    print("*** Running Enumeration ***")
        
    l_best =100
    for i in range(nits):
      K = np.array([np.random.choice(range(k_query),replace=True, p = pp) for pp in P])
      p = [ CHOICES[ch][K[ch]] for ch in range(len(CHOICES))]
      try:
        l = complex_diff(p)
      except: 
        l = 100
      if(l<l_best): 
        l_best=l
        print("Best score yet: {} with {}".format(l,p))
    exit()
  
  
  ## A Gaussian weighted exploration algorithm 
  if(True):
    l_best =100
    observations = []
    losses = []
    ## First carry out a random set of experiments
    for i in range(100):
      K = np.array([np.random.choice(range(k_query),replace=True, p = pp) for pp in P])
      p = [ CHOICES[ch][K[ch]] for ch in range(N_terms)]
      try:
        l = complex_diff(p)
      except: 
        l = 100
      if(l<l_best): 
        l_best=l
        print("Best score yet: {} with {}".format(l,p))
      observations.append(p)
      losses.append(l)
    
    for k in range(1):
      O = np.array(observations)
      L = np.array(losses)  
      ## Now for each parameter, derive a normal distribution 
      MS = np.array([ weighted_avg_and_std(np.transpose(O)[i], 1/L) for i in range(N_terms)])
      for i in range(100):
        p = [ np.random.normal(loc=m,scale = 2*s) for m,s in MS]
        try:
          l = complex_diff(p)
        except: 
          l = 100
        if(l<l_best): 
          l_best=l
          print("Best score yet: {} with {}".format(l,p))
        observations.append(p)
        losses.append(l)
    
    for k in range(1000):
      O = np.array(observations)
      L = np.array(losses)  
      ## Now for each parameter, derive a normal distribution 
      MS = np.array([ weighted_avg_and_std(np.transpose(O)[i], 1/L) for i in range(N_terms)])
    
      ## Consider the list of solutions weighted by the normals distributions
      PT_weights = [ [norm(loc=MS[qq][0],scale=MS[qq][1]).pdf(k) for k in CHOICES[qq]] for qq in range(len(MS))]
      PT_weights = [ a/np.sum(a) for a in PT_weights ]
      p = [ np.random.choice(CHOICES[i],p=PT_weights[i]) for i in range(len(CHOICES)) ]
      try:
        l = complex_diff(p)
      except: 
        l = 100
      if(l<l_best): 
        l_best=l
        print("Best score yet: {} with {}".format(l,p))
        print("Translates to: {} with {}".format(l,[reverse_dict[i] for i in p]))
      observations.append(p)
      losses.append(l)
  
  print("Best Params:",observations[np.argmin(losses)])

  ##
  #WITH THE BEST SCORE, search through the tree of values
  #for each parameter get say 5 values? 
  #Check the lists by eye? I.e.

  #"Basic Constant in [0,1,2,3]", then we can see if 0 is a bad suggestion?
  #"Blah Blah in ... ", 
  #From here we can run the above methods of filtering or a direct enumeration if the number of combinations is less than 2 million or so..
  #Run on a single data point, collect the best combinations and chec kfor multiple datapoints.

  #Consider a method to design splits i.e. as before on the filtering method
  #If we have two very high peaks, then reenumerate using those two values only!


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

