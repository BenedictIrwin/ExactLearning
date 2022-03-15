import numpy as np
import os
import sys
from exactlearning import wrap, plots
from scipy.optimize import minimize
from scipy.spatial import KDTree

pwd = os.getcwd()
string_list = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")

###########################
## Predefined complexity ##
###########################
## In order to best examine the spaces etc.
## We define sets of constants

## A dict of integers and rationals
rationals_dict = {}

keys = []
values = []
## Generate a dictionary of rationals
for i in range(20):
  for j in range(1,20):
    q = i/j
    st = "{}/{}".format(i,j)
    stlen = len(st)
    if(q not in values):
      values.append(q)
      keys.append(st)
      continue
    if(q in values):
      arg = np.argwhere(np.array(values)==q)[0][0]
      if(stlen<len(keys[arg])):
        values.pop(arg)
        keys.pop(arg)
        values.append(q)
        keys.append(st)

rationals_dict = {i:j for i,j in zip(keys,values)}

## A dict of only integers (useful for powers of gamma functions)
integers_dict = { "{}".format(i) : i for i in range(100) }

#non_zero_flexible_constants = [1,2,3]
#gamma_rational_constants = [-1,-2,-3,0,1,2,3,1/2,1/3,2/3] 
#hypergeometric_arguments = [-1,1,1/2,1/(2**2),(2**2)/(3**3),(3**3)/(4**4)] 

## A flexible list of possible prefactor type constants including gamma functions, roots and Pi
not_zero_dict = rationals_dict.copy()

## Sqrt of these constants for positive only BFGS varaibles
sqrt_not_zero_dict = { "sqrt({})".format(i) : np.sqrt(not_zero_dict[i]) for i in not_zero_dict.keys()}

## Hypergeometric arguments, should include 2**2/3**3 type numbers as well
hyp_arg_dict = rationals_dict.copy()

## For polynomial coeffieicnts (usually rationals)
poly_coeff_dict = rationals_dict.copy()


## A global list of constants, this is used for key matching
global_constants_dict = { "_notzero_" : not_zero_dict, "_sqrtnotzero_" : sqrt_not_zero_dict, "_gamma-rational_" : rationals_dict, "_hyp-arg_" : hyp_arg_dict, "poly-coeff" : poly_coeff_dict }

## This is a dict of trees for lookup of each type of constant
global_trees_dict = { i : KDTree([ [j] for j in global_constants_dict[i].values() ]) for i in global_constants_dict.keys() }


## Add a square to insist on positive log
constant_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "_notzero_", 
"logmoment" : "log(_sqrtnotzero_**2)" }

power_dict = {
        "n" : 1, 
        "req" : "", 
        "logreq" : "from numpy import log",
        "moment" : "_notzero_**_s_", 
        "logmoment" : "_s_*log(_sqrtnotzero_**2)" }

linear_gamma_dict = {
        "n" : 2, 
        "req" : "from scipy.special import gamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "gamma(_gamma-rational_ + _s_*_gamma-rational_)", 
        "logmoment" : "loggamma(_gamma-rational_ + _s_*_gamma-rational_)"}

scale_gamma_dict = {
        "n" : 1, 
        "req" : "from scipy.special import gamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "gamma(_s_*_gamma-rational_)", 
        "logmoment" : "loggamma(_s_*_gamma-rational_)"}

shift_gamma_dict = {
        "n" : 1, 
        "req" : "from scipy.special import gamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "gamma(_gamma-rational_ + _s_)", 
        "logmoment" : "loggamma(_gamma-rational_ + _s_)"}

neg_linear_gamma_dict = {
        "n" : 2, 
        "req" : "from scipy.special import rgamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "rgamma(_gamma-rational_ + _s_*_gamma-rational_)", 
        "logmoment" : "-loggamma(_gamma-rational_ + _s_*_gamma-rational_)"}

neg_scale_gamma_dict = {
        "n" : 1, 
        "req" : "from scipy.special import rgamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "rgamma(_s_*_gamma-rational_)", 
        "logmoment" : "-loggamma(_s_*_gamma-rational_)"}

neg_shift_gamma_dict = {
        "n" : 1, 
        "req" : "from scipy.special import rgamma", 
        "logreq" : "from scipy.special import loggamma", 
        "moment" : "rgamma(_gamma-rational_ + _s_)", 
        "logmoment" : "-loggamma(_gamma-rational_ + _s_)"}

## We only have one constant and let the prefactor absorb the scaling
P1_dict = {
        "n" : 1, 
        "req" : "", 
        "logreq" : "from numpy import log", 
        "moment" : "(1 + _s_*_poly-coeff_)", 
        "logmoment" : "log(1 + _s_*_poly-coeff_)"}

P2_dict = {
        "n" : 2, 
        "req" : "", 
        "logreq" : "from numpy import log", 
        "moment" : "(1 + _s_*_poly-coeff_ + _s_**2*_poly-coeff_)", 
        "logmoment" : "log(1 + _s_*_poly-coeff_ + _s_**2*_poly-coeff_)"}

twoFone_dict = {
        "n" : 7, 
        "req" : "from mpmath import hyp2f1", 
        "logreq" : "from mpmath import hyp2f1\nfrom numpy import log", 
        "moment" : "hyp2f1(_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_hyp-arg_)", 
        "logmoment" : "np.array([log(complex(hyp2f1(_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_hyp-arg_))) for ss in _s_])"}

oneFone_dict = {
        "n" : 5, 
        "req" : "from mpmath import hyp1f1", 
        "logreq" : "from mpmath import hyp1f1\nfrom numpy import log", 
        "moment" : "hyp1f1(_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_hyp-arg_)", 
        "logmoment" : "np.array([log(complex(hyp1f1(_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_hyp-arg_))) for ss in _s_])"}


## Somehow need to figure out the same parameters ?
pi_csc_pi_dict = {"n":2, "rules" : {"_A_" : "_rationals_", "_B_" : "_rationals_"}, "moment" : "(np.pi/_A_)*np.csc(np.pi*_s_/_A_ + np.pi/_B_)"}

## Consider a sum of the above units (this appears in elliptic K for example)

## possibly make this safe with mpmath gammaprod
norm_shift_gamma_dict = {"n" : 1,"rules" : {"_A_" : "_gamma-rational_"}, "req" : "from scipy.special import gamma, rgamma", "logreq" : "from scipy.special import loggamma","moment" : "gamma(_A_ + _s_)*rgamma(_A_)", "logmoment" : "loggamma(_A_ + _s_)-loggamma(_A_)" }  ## With rules

neg_norm_shift_gamma_dict = {"n" : 1,"rules" : {"_A_" : "_gamma-rational_"}, "req" : "from scipy.special import gamma, rgamma", "logreq" : "from scipy.special import loggamma","moment" : "rgamma(_A_ + _s_)*gamma(_A_)", "logmoment" : "-loggamma(_A_ + _s_)+loggamma(_A_)" }  ## With rules

norm_scale_gamma_dict = {"n" : 2,"rules" : {"_A_" : "_gamma-rational_", "_B_" : "_gamma-rational_"}, "req" : "from scipy.special import gamma, rgamma", "logreq" : "from scipy.special import loggamma","moment" : "_B_*gamma(_A_ + _B_*_s_)*rgamma(_A_)", "logmoment" : "log(_B_) + loggamma(_A_ + _B_*_s_)-loggamma(_A_)" }  ## With rules

neg_norm_scale_gamma_dict = {"n" : 2,"rules" : {"_A_" : "_gamma-rational_", "_B_" : "_gamma-rational_"}, "req" : "from scipy.special import gamma, rgamma", "logreq" : "from scipy.special import loggamma","moment" : "(1/_B_)*rgamma(_A_ + _B_*_s_)*gamma(_A_)", "logmoment" : "-log(_B_) - loggamma(_A_ + _B_*_s_)+loggamma(_A_)" }  ## With rules

## For two linear gammas?
linear_beta_dict = {"n": 2, "rules" : {"_A_"}, }

Erdelyi_G_function = {} ## With shift etc.
Higher_Polygamma_type_functions = {} ### 1,2,3,

## Window fitting ####################
## If we have an integral between [alpha,beta], this is the Mellin transform with Theta[x-alpha]*Theta[beta-x]
## as a window. For a hypergeometric result this gives
## (beta^s/s) (p+1)F(q+1)( ..., s; ..., 1+s; -beta) - (alpha^s/s) (p+1)F(q+1)( ..., s; ..., 1+s; -alpha)
## So by keeping the params the same and searching for a representation we can possibly identify the interior function
## with only a partial curve!
#######################################
## In addition to this, we can use differences in incomplete gamma functions?
################


### HyperU
### Generalised Hyper helper
### HyperComb()
### Special Polynomials? (i.e. Gegenbauer, Jacobi, order n+1/2 etc.)
### Zeta(s)


## FINISH
def generate_hyper_pFq(p,q):
  f_dict = {"0|1":"hyp0f1","1|1":"hyp1f1","1|2":"hyp1f2","2|0":"hyp2f0","2|1":"hyp2f1","2|2":"hyp2f2","2|3":"hyp2f3","3|2":"hyp3f2"}
  string_p = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(p)])+"]"
  string_q = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(q)])+"]"
  key = "{}|{}".format(p,q)
  if(key in f_dict.keys()):
    f_string = f_dict[key]
  ##  string = "{}({},{},_hyp-arg_)".format(f_string,str...)
  else: 
    f_string = "hyper"
  #return {"n" : 2*p+2*q+1, "req" : "from mpmath import {}".format(f_string), "logreq" : "from mpmath import {}\nfrom numpy import log".format(f_string), "moment" : "[complex({}) for ss in _s_]".format(string), "logmoment" : "np.array([log(complex({})) for ss in _s_])".format(string)}



## A helper function to enumerate many possible Meijer-G function combinations
def generate_meijerg_dict(a,b,c,d):
  string_a = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(a)])+"]"
  string_b = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(b)])+"]"
  string_c = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(c)])+"]"
  string_d = "["+",".join(["_gamma-rational_ + ss*_gamma-rational_" for i in range(d)])+"]"
  string = "meijerg(({},{}),({},{}),_hyp-arg_)".format(string_a,string_b,string_c,string_d)
  return {"n" : 2*a+2*b+2*c+2*d+1, "req" : "from mpmath import meijerg", "logreq" : "from mpmath import meijerg\nfrom numpy import log", "moment" : "[complex({}) for ss in _s_]".format(string), "logmoment" : "np.array([log(complex({})) for ss in _s_])".format(string)}

## Conversion between shorthand 'elements' and content
## Each entry represents a term which can be added
term_dict = {
"c" : constant_dict,
"c^s" : power_dict,
"linear-gamma" : linear_gamma_dict,
"scale-gamma" : scale_gamma_dict,
"shift-gamma" : shift_gamma_dict,
"neg-linear-gamma" : neg_linear_gamma_dict,
"neg-scale-gamma" : neg_scale_gamma_dict,
"neg-shift-gamma" : neg_shift_gamma_dict,
"P1" : P1_dict,
"P2" : P2_dict,
"2F1" : twoFone_dict
}

## Generate the Meijer-G dicts
for a in range(0,4):
  for b in range(0,4):
    for c in range(0,4):
      for d in range(0,4):
        term_dict["G-{}-{}-{}-{}".format(a,b,c,d)]= generate_meijerg_dict(a,b,c,d)

### EOF Predefined Complexity ###

## A helper function to assign pk parameter labels to constant spaces and vice versa
def parse_terms(terms):
  terms = terms.replace("_s_","")
  term_array = np.array(list(terms))
  underscores = np.argwhere( term_array == "_")
  undercount = len(underscores)
  if(undercount%2 != 0):
    print("Error! : Bad underscore count in {}".format(terms))
    exit()
  underscores = np.reshape(underscores,(undercount//2,2))
  strings = [terms[a:b+1] for a,b in underscores]
  param_type_dict = { strings[k] : "p{}".format(k) for k in range(len(strings)) }
  return strings , ["p{}".format(k) for k in range(undercount//2)]

## The exact learning estimator class
## Not designed to take a lot of information
class ExactEstimator:
  def __init__(self,tag, folder = ""):
    self.tag = tag
    self.folder = folder
    self.sample_mode = "first"
    self.fit_mode = "log"
    self.N_terms = None

    ## These are the object
    ## The fingerpint is the dict
    ## The function is a key that is a file (could eventually make a string) containing the function
    self.fingerprint = None
    self.function = None

    ## When functions are defined a record of their composition can be found here
    self.fingerprint_function_dict = {}
    self.function_terms_dict = {}

    ## Record the best results seen
    self.best_loss = np.inf
    self.best_params = None
    self.best_fingerprint = None
    self.best_function = None

    ## Record all results seen
    ## A dictionary of losses, hashed by
    self.results = {}

    ## Data
    self.s_values    = np.load("{}/s_values_{}.npy".format(folder,tag))
    self.sample_array = np.arange(self.s_values.shape[0])  ## This is for sampling from the samples
    self.moments     = np.load("{}/moments_{}.npy".format(folder,tag))
    self.logmoments  = np.log(self.moments)

    ## If the errors are to be found
    if(os.path.exists("{}/real_error_{}.npy".format(folder,tag))):
      self.real_error = np.load("real_error_{}.npy".format(tag))
    else:
      self.real_error = np.zeros(self.moments.shape)
    
    if(os.path.exists("{}/imag_error_{}.npy".format(folder,tag))):
      self.imag_error = np.load("imag_error_{}.npy".format(tag))
    else:
      self.imag_error = np.zeros(self.moments.shape)

    self.real_s = np.real(self.s_values)
    self.imag_s = np.imag(self.s_values)
    self.real_m = np.real(self.moments)
    self.imag_m = np.imag(self.moments)
    self.real_logm = np.real(self.logmoments)
    self.imag_logm = np.imag(self.logmoments)
    
    ## Calculate the error margins 
    real_log_upper = np.real(np.log(self.moments + self.real_error))
    real_log_lower = np.real(np.log(self.moments - self.real_error))
    imag_log_upper = np.imag(np.log(self.moments + self.imag_error))
    imag_log_lower = np.imag(np.log(self.moments - self.imag_error))

    ## The bounds to work with when fitting
    self.real_log_diff = real_log_upper - real_log_lower
    self.imag_log_diff = imag_log_upper - imag_log_lower
 

  ## Write a vectorized function to evaluate arbitary fingerprint
  ## The method is passed a dictionary to do this
  def write_function(self,key):
    print("Writing function: ",self.fingerprint,key)
    params = self.fingerprint.split(":")[3:]
    if(self.fit_mode == "log"):
      requirements = set( term_dict[term]["logreq"] for term in params )
      terms = ":".join([ term_dict[term]["logmoment"] for term in params])
    if(self.fit_mode == "normal"): 
      requirements = set( term_dict[term]["req"] for term in params )
      terms = ":".join([ term_dict[term]["moment"] for term in params]) 
    keys, values = parse_terms(terms) 
    self.function_terms_dict[key] = {values[i]:keys[i] for i in range(len(keys))}

    ## Writing a file that will rapidly fit the fingerprint 
    #with open(".\\Functions\\"+key+".py","w") as f:
    with open("./Functions/"+key+".py","w") as f:
      for req in requirements: f.write(req+"\n")
      f.write("from numpy import array, frompyfunc\n")
      f.write("def fp({}):\n".format(",".join(["p{}".format(i) for i in range(self.N_terms)])))
      iterate = 0
      if(self.fingerprint_N=="max"):
        S = self.s_values
      else:
        if(self.sample_mode == "random"):
          self.sample_array = np.random.choice(np.arange(self.fingerprint_N))
          S = self.s_values[self.sample_array]
        if(self.sample_mode == "first"): 
          self.sample_array = np.arange(0,self.fingerprint_N)
          S = self.s_values[self.sample_array]
      f.write("  S = array({})\n".format(list(S)))
      f.write("  ret = 0\n")
      for term in params:
        n_ = term_dict[term]["n"]
        if(self.fit_mode == 'log'): expression = term_dict[term]['logmoment']
        if(self.fit_mode == 'normal'): expression = term_dict[term]['moment']
        for subterm in range(n_):
          expression = expression.replace(keys[iterate+subterm],values[iterate+subterm],1)
        iterate+=n_
        f.write("  ret += {}\n".format(expression.replace("_s_","S")))
      f.write("  return ret\n")
      f.write("fingerprint = frompyfunc(fp,{},1)\n".format(self.N_terms))

  def set_fingerprint(self,fp_hash):
    self.fingerprint = fp_hash

    ## Add a list to collect results dictionaries under this has
    if(self.fingerprint not in self.results): self.results[self.fingerprint] = []
    
    hash_list = fp_hash.split(":")
    name = hash_list[0]
    hash_mode = hash_list[1]
    self.fingerprint_N = hash_list[2]
    if(self.fingerprint_N!="max"): self.fingerprint_N = int(self.fingerprint_N)
    self.sample_mode = hash_mode ## Important for random samples
    count = [ term_dict[term]["n"] for term in hash_list[3:]]
    self.N_terms = np.sum(count) 
    ## If we have seen this fingerprint before, just load up the previous file
    if(self.fingerprint in self.fingerprint_function_dict and hash_mode == 'first'):
      print(self.function)
      print(self.fingerprint_function_dict)
      self.function = self.fingerprint_function_dict[self.fingerprint]
      print(self.function)
      #with open(".\\Functions\\{}.py".format(self.function),"r") as f: flines = f.readlines()
      with open("./Functions/{}.py".format(self.function),"r") as f: flines = f.readlines()
      if('fingerprint' in globals().keys()): del globals()['fingerprint']
      exec("".join(flines),globals())
      print("Function is reset to {}!".format(self.function))
      return

    ## Otherwise write a vectorized loss function in terms of the input data etc. 
    key = "".join([np.random.choice(string_list) for i in range(10)])
    self.write_function(key)
    if('fingerprint' in globals().keys()): del globals()['fingerprint']
    #with open(".\\Functions\\{}.py".format(key),"r") as f: flines = f.readlines()
    with open("./Functions/{}.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    self.fingerprint_function_dict[self.fingerprint] = key
    self.function = key

  ## A gradient based approach to get the optimial parameters for a given fingerprint
  def BFGS(self,p0=None):
    #if(p0==None): p0=np.random.random(self.N_terms)
    if(p0==None): p0=np.random.uniform(low=-10,high=10,size=self.N_terms)
    if(self.fit_mode=="log"):
      f1 = self.real_log_loss
      f2 = self.complex_log_loss
    if(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()

    '''  
    print("TEST OVERRIDE!")
    '''
    #p0 = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
    #print("DEBUG_init f1: ",p0, "loss", f1(p0))
    #print("DEBUG_init f2: ",p0, "loss", f2(p0))
    #exit()
    '''
    plots(self.s_values,self.logmoments,fingerprint(*p0))
    test = np.array(self.logmoments)
    pred = np.array(fingerprint(*p0))[0]
    print(test.shape)
    print(pred.shape)

    print(test[:10])
    print(pred[:10])
    print(np.real(test[:10]))
    print(np.real(pred[:10]))
    print(np.imag(test[:10]))
    print(np.imag(pred[:10]))

    from sklearn.metrics import r2_score

    print(r2_score(np.real(test),np.real(pred)))
    print(f1(p0))

    exit()
    '''

    res = minimize(f1, x0=p0, method = "BFGS", tol = 1e-6)
    print("DEBUG_v1: ",res.x, "loss", f1(res.x)) 
    ##plots(self.s_values,self.logmoments,fingerprint(*res.x))
    #print("Params: ", res.x)
    #print("Loss: ", f1(res.x))
    res = minimize(f2, x0=res.x, method = "BFGS", tol = 1e-8)
    print("DEBUG_v2: ",res.x, "loss", f2(res.x)) 
    ##plots(self.s_values,self.logmoments,fingerprint(*res.x))
    loss = f2(res.x)
    #print("Final Solution: ",res.x)
    #print("Loss: ", loss)
    ## Add this result to the list
    self.register(res.x,loss)
    return res.x, loss

  ## Store the results
  def register(self,params,loss):
    ## Best Solution (eventually have top k solutions BPQ)
    if(loss < self.best_loss):
      print("New record solution! {}\n{}".format(loss,self.fingerprint))
      self.best_loss = loss
      self.best_params = params
      self.best_fingerprint = self.fingerprint
      self.best_function = self.function

    descriptors = self.descriptors_from_fingerprint(self.fingerprint)
    self.results[self.fingerprint].append({"params" : params, "loss" : loss, "descriptors" : descriptors})

  ## Convert params to a unifed set of descriptors
  ## This may be useful to train models that predict loss via function composition
  ## Consider a Free-Wilson type analysis
  def descriptors_from_fingerprint(self,fingerprint):
    fp_list = fingerprint.split(":")[3:]
    descriptors = []
    elements = ["c","c^s","linear-gamma","scale-gamma","shift-gamma"]
    descriptors = [ fp_list.count(e) for e in elements ]
    return descriptors

  ## Vectorised difference function
  def real_log_loss(self,p): 
    A = fingerprint(*p)[0]
    #print(self.sample_array)
    #print(self.sample_array.shape)

    #B = np.abs(np.real(A)-self.real_logm[self.sample_array])
    #B = np.maximum(0.0,B-self.real_log_diff[self.sample_array])

    B = np.abs(np.real(A)-self.real_logm)
    B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)

  ## Vectorised difference function
  def complex_log_loss(self,p):
    #print("WARNING SELF.sample_array seems broken!")
    #exit()
    A = fingerprint(*p)
    #B = np.abs(np.real(A)-self.real_logm[self.sample_array])
    #B = np.maximum(0.0,B-self.real_log_diff[self.sample_array])
    #C = np.abs(wrap(np.imag(A)-self.imag_logm[self.sample_array]))
    #C = np.maximum(0.0,C-self.imag_log_diff[self.sample_array])
    B = np.abs(np.real(A)-self.real_logm)
    B = np.maximum(0.0,B-self.real_log_diff)
    C = np.abs(wrap(np.imag(A)-self.imag_logm))
    C = np.maximum(0.0,C-self.imag_log_diff)
    return np.mean(B+C)

  def set_mode(self,mode): 
    if(mode not in ["log","normal"]):
      print("Error: valid mode choices are 'log' or 'normal' for logmoments or moments respectively!")
      exit()
    self.fit_mode = mode

  def summarise(self):
    results = self.results
    keys = self.results.keys()
    losses = [[j["loss"] for j in results[key]] for key in keys]
    minimum_loss = np.amin(losses,axis=1)
    for i,j in zip(keys,minimum_loss):
      print(i,j)

  ## A function which suggests possible closed forms
  def speculate(self):
    print("Best result is: ")
    print("- Fingerprint: ",self.best_fingerprint)
    print("- Parameters: ",self.best_params)
    print("- Loss: ",self.best_loss)
    terms_list = self.function_terms_dict[self.best_function]

    print("Best Function ID: ",self.best_function)
    print("Log Space Function Expression: ~~~~")
    with open("Functions/{}.py".format(self.best_function),"r") as ff:
      flines = [i.strip() for i in ff.readlines()]
      flines = [i for i in flines if "ret " in i]
      for i in flines: 
        print(i)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")

    ## For each parameter get the correct list:KDTree pair
    for par,value in zip(terms_list.keys(),self.best_params):
      term_type = terms_list[par]
      print("*** ",par,":",term_type," ***")
      ## Could do a search within  tolerance or just do nearest neighbours
      distances, IDs = global_trees_dict[term_type].query([np.abs(value)],k=4)
      dict_IDs = [ list(global_constants_dict[term_type].keys())[i] for i in IDs]
      for i, delta in zip(dict_IDs,distances):
        print(value,"~","{}".format("-" if value < 0 else "")+i,u" (\u0394 = {})".format(delta).encode('utf-8'))
      
      print("*** ","~~~"," ***") 

    p0 = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
    print("Ideal (test)--> ", p0)

## A function that helps
def gen_fpdict(parlist,N='max',mode='first',name=''):
  hash_string = ":".join(sorted(parlist))
  #print(kwargs)
  #if("name" not in kwargs.keys()): name = ''
  #else: name = kwargs["name"]
  #if("mode" not in kwargs.keys()): mode = 'first'
  #else: mode = kwargs["mode"]
  #if("N" not in kwargs.keys()): N = 'max'
  #else: N = kwargs["N"]
  if(mode not in['first','random']):  
    print("Error: gen_fpdict mode can only be 'first' or 'random' (or blank===first)!")
    exit()
  hash_string = ":".join([name,mode,str(N),hash_string])
  return hash_string

#########################################
## Start the code here pointing to a file
EE = ExactEstimator("Chi_Distribution", folder = "Chi_Distribution")

fps=[
#gen_fpdict(['c','linear-gamma']),
#gen_fpdict(['c^s','linear-gamma']),
gen_fpdict(['c','c^s','linear-gamma']),
#gen_fpdict(['c','c^s','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','linear-gamma','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','P1']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','P2']),
#gen_fpdict(['c','c^s','2F1'],N=4),
#gen_fpdict(['c','c^s','2F1','linear-gamma'],N=4),
#gen_fpdict(['c','c^s','2F1','linear-gamma','neg-linear-gamma'],N=4)
]
#fp_x = gen_fpdict(['c','c^s','G-1-0-1-1'])   ### Look up how to do kwargs properly...

## Looping over fingerprints to see which is best
for k in fps:
  print("Setting Fingerprint: ",k)
  EE.set_fingerprint(k)
  n_bfgs = 30
  for i in range(n_bfgs):
    EE.BFGS()
    print("{}%".format(100*(i+1)/n_bfgs),flush=True)

print("Completed!")
EE.speculate()

EE.set_fingerprint(EE.best_fingerprint)

print(EE.best_params)
#optimal_params = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
#plots(EE.s_values,EE.logmoments,fingerprint(*optimal_params))
plots(EE.s_values,EE.logmoments,fingerprint(*EE.best_params))

##print(EE.__dict__)
print(EE.results)

exit()






exit()
EE.set_fingerprint(fp_2)
for i in range(10): EE.BFGS()
EE.set_fingerprint(fp_3)
for i in range(10): EE.BFGS()

res = EE.results
print(res)

EE.summarise()

EE.set_mode('log')






