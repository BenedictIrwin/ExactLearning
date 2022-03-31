import numpy as np
import os
import sys
from exactlearning import wrap
from scipy.optimize import minimize



pwd = os.getcwd()
if(not os.path.isdir("Functions")): os.mkdir("Functions")
sys.path.insert(1, pwd+"\\Functions")

string_list = "a b c d e f g h i j k l m n o p".split(" ")

## Predefined complexity - In order to best examine the spaces etc.
## We define sets of constants
constant_dict = {
        "_not-zero_" : [1,2,3], ## Actually use a preset very complex here
        "_sqrt-not-zero_" : np.sqrt([1,2,3]), 
        "_gamma-rational_" : [1,2,3,1/2,1/3,2/3]
}

## Add a square to insist on positive log
constant_dict = {
"n" : 1, 
"req" : None, 
"logreq" : "from numpy import log", 
"moment" : "_notzero_", 
"logmoment" : "log(_sqrtnotzero_**2)" }

power_dict = {
        "n" : 1, 
        "req" : None, 
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

twoFone_dict = {
        "n" : 7, 
        "req" : "from mpmath import hyp2f1", 
        "logreq" : "from mpmath import hyp2f1\nfrom numpy import log", 
        "moment" : "hyp2f1(_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_hyp-arg_)", 
        "logmoment" : "[log(complex(hyp2f1(_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_hyp-arg_))) for ss in _s_]"}

oneFone_dict = {
        "n" : 5, 
        "req" : "from mpmath import hyp1f1", 
        "logreq" : "from mpmath import hyp1f1\nfrom numpy import log", 
        "moment" : "hyp1f1(_gamma-rational_ + _s_*_gamma-rational_,_gamma-rational_ + _s_*_gamma-rational_,_hyp-arg_)", 
        "logmoment" : "[log(complex(hyp1f1(_gamma-rational_ + ss*_gamma-rational_,_gamma-rational_ + ss*_gamma-rational_,_hyp-arg_))) for ss in _s_]"}


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
"2F1" : twoFone_dict
}

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
  def __init__(self,tag):
    self.tag = tag
    self.sample_mode = "first"
    self.fit_mode = "log"
    self.N_terms = None

    ## These are the object
    ## The fingerpint is the dict
    ## The function is a key that is a file containing the function
    self.fingerprint = None
    self.function = None

    self.fingerprint_function_dict = {}

    ## Record the best results seen
    self.best_loss = np.inf
    self.best_params = None
    self.best_fingerprint = None

    ## Data
    self.s_values    = np.load("s_values_{}.npy".format(tag))
    self.num_samples = self.s_values.shape[0]
    self.sample_array = np.arange(self.num_samples)  ## This is for sampling from the samples
    self.moments     = np.load("moments_{}.npy".format(tag))
    self.logmoments  = np.log(self.moments)
    self.real_error = np.load("real_error_{}.npy".format(tag))
    self.imag_error = np.load("imag_error_{}.npy".format(tag))
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
  def write_function(self,key, N="max"):
    print(self.fingerprint,key)
    name = self.fingerprint["name"]
    params = self.fingerprint["parameters"]
    if(self.fit_mode == "log"):
      requirements = set( term_dict[term]["logreq"] for term in params )
      terms = ":".join([ term_dict[term]["logmoment"] for term in params])
    if(self.fit_mode == "normal"): 
      requirements = set( term_dict[term]["req"] for term in params )
      terms = ":".join([ term_dict[term]["moment"] for term in params]) 
    keys, values = parse_terms(terms)
    count = [ term_dict[term]["n"] for term in params]
    self.N_terms = np.sum(count) 
    
    ## Writing a file that will rapidly fit the fingerprint 
    with open(pwd+"\\Functions\\"+key+".py","w") as f:
      for req in requirements: f.write(req+"\n")
      f.write("from numpy import array, frompyfunc\n")
      f.write("def fp({}):\n".format(",".join(["p{}".format(i) for i in range(self.N_terms)])))
      iterate = 0
      if(N=="max"):
        S = self.s_values
      else:
        if(self.sample_mode == "random"):
          self.sample_array = np.random.choice(np.arange(self.num_samples))
          S = self.s_values[self.sample_array]
        if(self.sample_mode == "first"): 
          self.sample_array = np.arange(0,N)
          S = self.s_values[self.sample_array]
      f.write("  S = array({})\n".format(list(S)))
      f.write("  ret = 0\n")
      for term in params:
        n_ = term_dict[term]["n"]
        if(self.fit_mode == 'log'): expression = term_dict[term]['logmoment']
        if(self.fit_mode == 'normal'): expression = term_dict[term]['moment']
        for subterm in range(n_):
          expression = expression.replace(keys[iterate+subterm],values[iterate+subterm],1)
        iterate+=1
        f.write("  ret += {}\n".format(expression.replace("_s_","S")))
      f.write("  return ret\n")
      f.write("fingerprint = frompyfunc(fp,{},1)\n".format(self.N_terms))
      f.write("print('End of {}')".format(name))      

  ## We consider N?
  def set_fingerprint(self,fp_dict, N = "max"):
    self.fingerprint = fp_dict
    print("Setting fingerprint to: {}".format(self.fingerprint["name"]))
    
    ## If we have seen this fingerprint before, just load up the previous file
    if(self.fingerprint in self.fingerprint_function_dict.values()):
      self.function = self.fingerprint_fuction_dict[self.fingerprint]
      exec("from {} import fingerprint".format(key,globas()))
      print("Function is {}".format(self.function))
      return

    ## Otherwise write a vectorized loss function in terms of the input data etc. 
    key = "".join([np.random.choice(string_list) for i in range(10)])
    print(self.write_function(key, N=N))
    exec("from {} import fingerprint".format(key),globals())
    self.fingerprint_function_dict[self.fingerprint] = key
    self.function = key

  ## A gradient based approach to get the optimial parameters for a given fingerprint
  def BFGS(self,p0=None):
    if(p0==None): p0=np.random.random(self.N_terms)
    if(self.fit_mode=="log"):
      f1 = self.real_log_loss
      f2 = self.complex_log_loss
    if(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()
    res = minimize(f1, x0=p0, method = "BFGS", tol = 1e-6)
    print("Params: ", res.x)
    print("Loss: ", f1(res.x))
    res = minimize(f2, x0=res.x, method = "BFGS", tol = 1e-8)
    loss = f2(res.x)
    print("Final Solution: ",res.x)
    print("Loss: ", loss)

    ## Record Solution
    if(loss < self.best_loss):
      print("New record solution! {}".format(loss))
      self.best_loss = loss
      self.best_params = res.x
      self.best_fingerprint = self.fingerprint

    return res.x, loss

  ## Vectorised difference function
  def real_log_loss(self,p): 
    A = fingerprint(*p)
    B = np.abs(np.real(A)-self.real_logm[self.sample_array])
    B = np.maximum(0.0,B-self.real_log_diff[self.sample_array])
    return np.mean(B)

  ## Vectorised difference function
  def complex_log_loss(self,p):
    A = fingerprint(*p)
    B = np.abs(np.real(A)-self.real_logm[self.sample_array])
    B = np.maximum(0.0,B-self.real_log_diff[self.sample_array])
    C = np.abs(wrap(np.imag(A)-self.imag_logm[self.sample_array]))
    C = np.maximum(0.0,C-self.imag_log_diff[self.sample_array])
    return np.mean(B+C)

  def set_mode(self,mode): 
    if(mode not in ["log","normal"]):
      print("Error: valid mode choices are 'log' or 'normal' for logmoments or moments respectively!")
      exit()
    self.fit_mode = mode


#########################################
## Start the code here pointing to a file
EE = ExactEstimator("ExpTest")

fp_1 = {"name":"constant+power-s+linear-gamma","parameters" : ["c","c^s","linear-gamma"]}
fp_2 = {"name":"2F1","parameters" : ["c","c^s","2F1"]}
fp_3 = {"name":"Cs-scale-gamma","parameters" : ["c","c^s","scale-gamma"]}
fp_4 = {"name":"Cs-shift-gamma","parameters" : ["c","c^s","shift-gamma"]}

EE.set_fingerprint(fp_3,N=10)
for i in range(10): EE.BFGS()
EE.set_mode('log')






