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


### Notes for extensions... 

## Where we have phi(s) = C K^s P_k(s)/Q_l(s) * Gamma()/Gamma() product
## With P and Q as polynomials (i.e. Quantum Mechanics Examples)
## We can isolate a quantity

## phi'(s)/phi(s) + Q'(s)/Q(s) - P'(s)/P(s) = log(K) + Sum( digamma(...) )
## I.e. we can insert the polynomial terms to the other side as well

## This allows the logd estimator to be used for polynomial terms

## In general... any new term added by product to the fingerprint i.e. zeta(s)Gamma(s)
## When the logarithmic derivative is taken, we get the extra term, so this can be very general
## phi'(s)/phi(s) + zeta'(s)/zeta(s)

###################################################################################################
## For dddq we have that 2 (dq/q)**3 - 3 (dq/q) * (ddq/q) + (dddq/q) ~ Sum b^3 tetragamma(a + b s)#
###################################################################################################

## For polynomials we duplicate the higher order term as well..

#############################################################################
## We need to consider how the concept of order works for higher dimensions!
## Gamma(s1 + s2)Gamma[s1]Gamma[s2] -->
## phi(s1,s2), phi[',](s1,s2), phi[,'](s1,s2), phi[','](s1,s2)
## D[ Log[phi[s1,s2]], s1] // FullSimplify
## PolyGamma[0, s1] + PolyGamma[0, s1 + s2]
## D[ Log[phi[s1,s2]], s2] // FullSimplify
## PolyGamma[0, s2] + PolyGamma[0, s1 + s2]
##
## This is great at isolating parts
## D[ Log[phi[s1,s2]], s1,s2] // FullSimplify
## PolyGamma[1, s1 + s2]
#############################################################################

## Consider MGF, log-convex.
## Consider bounds on the variables going into the gamma functions thus.
## Gamma(a + b s), tested for all s... if the Re(a + b s) is <= 0 then there is a problem




## This might need to be improved?
def deal(p0,states):
  p0 = list(p0)
  itr = 0
  mx = len(states)
  vec = []
  while(itr<mx):
    if(states[itr]):vec.append(0)
    else: vec.append(p0.pop(0))
    itr+=1
  return np.array(vec)

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
global_constants_dict = { "_notzero_" : not_zero_dict, "_sqrtnotzero_" : sqrt_not_zero_dict, "_gamma-rational_" : rationals_dict, "_hyp-arg_" : hyp_arg_dict, "_poly-coeff_" : poly_coeff_dict }

## This is a dict of trees for lookup of each type of constant
global_trees_dict = { i : KDTree([ [j] for j in global_constants_dict[i].values() ]) for i in global_constants_dict.keys() }


## Add a square to insist on positive log
constant_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "_notzero_", 
"logmoment" : "log(_sqrtnotzero_**2)",
"logderivative" : "0",
"logderivativereq" : "",
"logderivative2" : "0",
"logderivative2req" : "",
"terms" : ["_sqrtnotzero_"],
"MMA" : "_sqrtnotzero_**2",
"sympy" : "_sqrtnotzero_**2",
}

s_dict = {
"n" : 0, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "_s_", 
"logmoment" : "log(_s_)",
"logderivative" : "1/_s_",
"logderivativereq" : "",
"logderivative2" : "-1/(_s_**2)",
"logderivative2req" : "",
"terms" : [],
"MMA" : "_s_",
"sympy" : "_s_",
}

s_power_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "_s_**_poly-coeff_", 
"logmoment" : "_poly-coeff_*log(_s_)",
"logderivative" : "_poly-coeff_/_s_",
"logderivativereq" : "",
"logderivative2" : "-_poly-coeff_/(_s_**2)",
"logderivative2req" : "",
"terms" : ["_poly-coeff_"],
"MMA" : "_s_**_poly-coeff_",
"sympy" : "_s_**_poly-coeff_",
}

neg_s_dict = {
"n" : 0, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "(1/_s_)", 
"logmoment" : "-log(_s_)",
"logderivative" : "-1/_s_",
"logderivativereq" : "",
"logderivative2" : "1/(_s_**2)",
"logderivative2req" : "",
"terms" : [],
"MMA" : "(1/_s_)",
"sympy" : "(1/_s_)",
}

neg_s_power_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "(1/(_s_**_poly-coeff_))", 
"logmoment" : "-_poly-coeff_*log(_s_)",
"logderivative" : "-_poly-coeff_/_s_",
"logderivativereq" : "",
"logderivative2" : "_poly-coeff_/(_s_**2)",
"logderivative2req" : "",
"terms" : ["_poly-coeff_"],
"MMA" : "(1/(_s_^_poly-coeff_))",
"sympy" : "(1/(_s_**_poly-coeff_))",
}

power_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log",
"moment" : "_notzero_**_s_", 
"logmoment" : "_s_*log(_sqrtnotzero_**2)",
"logderivative" : "log(_sqrtnotzero_**2)",
"logderivativereq" : "from numpy import log",
"logderivative2" : "0",
"logderivative2req" : "",
"terms" : ["_sqrtnotzero_"],
"MMA" : "_sqrtnotzero_^(2*_s_)",
"sympy" : "_sqrtnotzero_**(2*_s_)",
}

linear_gamma_dict = {
"n" : 2, 
"req" : "from scipy.special import gamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "gamma(_gamma-rational_ + _s_*_gamma-rational_)", 
"logmoment" : "loggamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative" : "_VAR2_*digamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "_VAR2_**2*np.trigamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "Gamma[_gamma-rational_ + _s_*_gamma-rational_]",
"sympy" : "gamma(_gamma-rational_ + _s_*_gamma-rational_)",
"terms" : ["_gamma-rational_","_gamma-rational_"]
}

alt_linear_gamma_dict = {
"n" : 2, 
"req" : "from scipy.special import gamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "gamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)", 
"logmoment" : "loggamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative" : "_VAR2_*digamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "_VAR2_**2*np.trigamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "Gamma[_VAR2_*_gamma-rational_ + _s_*_gamma-rational_]",
"sympy" : "gamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"terms" : ["_gamma-rational_","_gamma-rational_"]
}

scale_gamma_dict = {
"n" : 1, 
"req" : "from scipy.special import gamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "gamma(_s_*_gamma-rational_)", 
"logmoment" : "loggamma(_s_*_gamma-rational_)",
"terms" : ["_gamma-rational_"],
"logderivative" : "_VAR1_*digamma(_s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "_VAR1_**2*np.trigamma(_s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "Gamma[_s_*_gamma-rational_]",
"sympy" : "gamma(_s_*_gamma-rational_)",
}

shift_gamma_dict = {
"n" : 1, 
"req" : "from scipy.special import gamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "gamma(_gamma-rational_ + _s_)", 
"logmoment" : "loggamma(_gamma-rational_ + _s_)",
"logderivative" : "digamma(_gamma-rational_ + _s_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "np.trigamma(_gamma-rational_ + _s_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "Gamma[_gamma-rational_ + _s_]",
"sympy" : "gamma(_gamma-rational_ + _s_)",
"terms" : ["_gamma-rational_"]
}

neg_linear_gamma_dict = {
"n" : 2, 
"req" : "from scipy.special import rgamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "rgamma(_gamma-rational_ + _s_*_gamma-rational_)", 
"logmoment" : "-loggamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative" : "-_VAR2_*digamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "-_VAR2_**2*np.trigamma(_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"terms" : ["_gamma-rational_","_gamma-rational_"],
"MMA" : "(Gamma[_gamma-rational_ + _s_*_gamma-rational_])^(-1)",
"sympy" : "(gamma(_gamma-rational_ + _s_*_gamma-rational_))**(-1)",
}

alt_neg_linear_gamma_dict = {
"n" : 2, 
"req" : "from scipy.special import gamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "rgamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)", 
"logmoment" : "-loggamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative" : "-_VAR2_*digamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "-_VAR2_**2*np.trigamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "(Gamma[_VAR2_*_gamma-rational_ + _s_*_gamma-rational_])^(-1)",
"sympy" : "(gamma(_VAR2_*_gamma-rational_ + _s_*_gamma-rational_))**(-1)",
"terms" : ["_gamma-rational_","_gamma-rational_"]
}

neg_scale_gamma_dict = {
"n" : 1, 
"req" : "from scipy.special import rgamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "rgamma(_s_*_gamma-rational_)", 
"logmoment" : "-loggamma(_s_*_gamma-rational_)",
"terms" : ["_gamma-rational_"],
"logderivative" : "-_VAR1_*digamma(_s_*_gamma-rational_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "-_VAR1_**2*np.trigamma(_s_*_gamma-rational_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "(Gamma[_s_*_gamma-rational_])^(-1)",
"sympy" : "(gamma(_s_*_gamma-rational_))**(-1)",
}

neg_shift_gamma_dict = {
"n" : 1, 
"req" : "from scipy.special import rgamma", 
"logreq" : "from scipy.special import loggamma", 
"moment" : "rgamma(_gamma-rational_ + _s_)", 
"logmoment" : "-loggamma(_gamma-rational_ + _s_)",
"logderivative" : "-digamma(_gamma-rational_ + _s_)",
"logderivativereq" : "from scipy.special import digamma",
"logderivative2" : "-np.trigamma(_gamma-rational_ + _s_)",
"logderivative2req" : "from AdvancedFunctions import *",
"MMA" : "(Gamma[_gamma-rational_ + _s_])^(-1)",
"sympy" : "(gamma(_gamma-rational_ + _s_))**(-1)",
"terms" : ["_gamma-rational_"]
}
## We only have one constant and let the prefactor absorb the scaling
P1_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "(1.0 + _s_*_poly-coeff_)", 
"logmoment" : "log(1.0 + _s_*_poly-coeff_)",
"logderivative" : "_VAR1_/(1.0 + _s_*_poly-coeff_)",
"logderivativereq" : "",
"logderivative2" : "-_VAR1_**2/(1.0 + _s_*_poly-coeff_)**2",
"logderivative2req" : "",
"MMA" : "(1 + _s_*_poly-coeff_)",
"sympy" : "(1 + _s_*_poly-coeff_)",
"terms" : ["_poly-coeff_"]
}

neg_P1_dict = {
"n" : 1, 
"req" : "", 
"logreq" : "from numpy import log", 
"moment" : "1.0/(1.0 + _s_*_poly-coeff_)", 
"logmoment" : "-log(1.0 + _s_*_poly-coeff_)",
"logderivative" : "-_VAR1_/(1.0 + _s_*_poly-coeff_)",
"logderivativereq" : "",
"logderivative2" : "_VAR1_**2/(1.0 + _s_*_poly-coeff_)**2",
"logderivative2req" : "",
"MMA" : "(1 + _s_*_poly-coeff_)^(-1)",
"sympy" : "(1 + _s_*_poly-coeff_)**(-1)",
"terms" : ["_poly-coeff_"]
}

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
"s" : s_dict,
"1/s" : neg_s_dict,
"s^a" : s_power_dict,
"1/s^a" : neg_s_power_dict,
"linear-gamma" : linear_gamma_dict,
"alt-linear-gamma" : alt_linear_gamma_dict,
"scale-gamma" : scale_gamma_dict,
"shift-gamma" : shift_gamma_dict,
"neg-linear-gamma" : neg_linear_gamma_dict,
"alt-neg-linear-gamma" : alt_neg_linear_gamma_dict,
"neg-scale-gamma" : neg_scale_gamma_dict,
"neg-shift-gamma" : neg_shift_gamma_dict,
"P1" : P1_dict,
"neg-P1" : neg_P1_dict,
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
  terms = terms.replace("_VAR1_","")
  terms = terms.replace("_VAR2_","")
  if("_VAR3_" in terms):
    print("PARSE TERMS ERROR!, Need to upgrade this code")
    exit()
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

    self.BFGS_derivative_order = 0

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

    ## Get the dimension of the s_values
    if(len(self.s_values.shape) == 1):
      self.n_s_dims = 1
    else:
      if(len(self.s_values.shape) > 2):
        print("Error: Tensor values moments detected! dim s = {}".format(self.s_values.shape))
        exit()
      self.n_s_dims = self.s_values.shape[-1]


    ## Consider deleting
    self.sample_array = np.arange(self.s_values.shape[0])  ## This is for sampling from the samples

    self.moments     = np.load("{}/moments_{}.npy".format(folder,tag))
    #self.logmoments  = np.log(self.moments)

    ## Load in the ratio dq/q which is expected to be a sum of digamma functions
    self.ratio = np.load("{}/logderivative_{}.npy".format(folder,tag))
    self.ratio2 = np.load("{}/logderivative2_{}.npy".format(folder,tag))

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
    #self.real_logm = np.real(self.logmoments)
    #self.imag_logm = np.imag(self.logmoments)
    
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
      logderivative_requirements = set( term_dict[term]["logderivativereq"] for term in params )
      logderivative_terms = ":".join([ term_dict[term]["logderivative"] for term in params])
      logderivative2_requirements = set( term_dict[term]["logderivative2req"] for term in params )
      logderivative2_terms = ":".join([ term_dict[term]["logderivative2"] for term in params])
      print(logderivative_terms)
      print(logderivative_requirements)
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

      #print(S)
      #print("{}".format(list(S)))
      #exit()

      f.write("  ret = 0\n")
      for term in params:
        n_ = term_dict[term]["n"]
        if(self.fit_mode == 'log'): expression = term_dict[term]['logmoment']
        if(self.fit_mode == 'normal'): expression = term_dict[term]['moment']
        for subterm in range(n_):
          expression = expression.replace(keys[iterate+subterm],values[iterate+subterm],1)
          subterm_var = "_VAR"+str(subterm+1)+"_"
          expression = expression.replace(subterm_var,values[iterate+subterm])
        iterate+=n_
        f.write("  ret += {}\n".format(expression.replace("_s_","S")))
        #f.write("  ret = np.add(ret,{},out=ret,casting='unsafe')\n".format(expression.replace("_s_","S")))
      f.write("  return ret\n")
      f.write("fingerprint = frompyfunc(fp,{},1)\n".format(self.N_terms))

    ## Also write a log derivate function
    if(True):
      with open("./Functions/"+key+"_logderivative.py","w") as f:
        for req in logderivative_requirements: f.write(req+"\n")
        f.write("from numpy import array, frompyfunc\n")
        ### N.B. c term will do nothing
        f.write("def fp({}):\n".format(",".join(["p{}".format(i) for i in range(self.N_terms)])))
        iterate = 0
        if(self.fingerprint_N=="max"):
          S = self.s_values
        #else:
        #  if(self.sample_mode == "random"):
        #    self.sample_array = np.random.choice(np.arange(self.fingerprint_N))
        #    S = self.s_values[self.sample_array]
        #  if(self.sample_mode == "first"): 
        #    self.sample_array = np.arange(0,self.fingerprint_N)
        #    S = self.s_values[self.sample_array]
        f.write("  S = array({})\n".format(list(S)))
        f.write("  ret = 0\n")
        for term in params:
          n_ = term_dict[term]["n"]
          #if(self.fit_mode == 'log'): expression = term_dict[term]['logmoment']
          #if(self.fit_mode == 'normal'): expression = term_dict[term]['moment']
          print("n",n_)
          expression = term_dict[term]['logderivative']
          print(expression)
          for subterm in range(n_):
            expression = expression.replace(keys[iterate+subterm],values[iterate+subterm],1)
            ## Add an exception to reference the already used variables using a keyword
            #print(subterm_var, values[iterate+subterm]) 
            #if(subterm_var in expression): (If it's not there it won't be replaced)
            subterm_var = "_VAR"+str(subterm+1)+"_"
            expression = expression.replace(subterm_var,values[iterate+subterm])
          iterate+=n_
          f.write("  ret += {}\n".format(expression.replace("_s_","S")))
          #f.write("  ret = np.add(ret,{},out=ret,casting='unsafe')\n".format(expression.replace("_s_","S")))
        f.write("  return ret\n")
        f.write("logderivative = frompyfunc(fp,{},1)\n".format(self.N_terms))
    
    ## Also write a log derivate 2 function
    if(True):
      with open("./Functions/"+key+"_logderivative2.py","w") as f:
        for req in logderivative2_requirements: f.write(req+"\n")
        f.write("from numpy import array, frompyfunc\n")
        ### N.B. c term will do nothing
        f.write("def fp({}):\n".format(",".join(["p{}".format(i) for i in range(self.N_terms)])))
        iterate = 0
        if(self.fingerprint_N=="max"):
          S = self.s_values
        #else:
        #  if(self.sample_mode == "random"):
        #    self.sample_array = np.random.choice(np.arange(self.fingerprint_N))
        #    S = self.s_values[self.sample_array]
        #  if(self.sample_mode == "first"): 
        #    self.sample_array = np.arange(0,self.fingerprint_N)
        #    S = self.s_values[self.sample_array]
        f.write("  S = array({})\n".format(list(S)))
        f.write("  ret = 0\n")
        for term in params:
          n_ = term_dict[term]["n"]
          #if(self.fit_mode == 'log'): expression = term_dict[term]['logmoment']
          #if(self.fit_mode == 'normal'): expression = term_dict[term]['moment']
          expression = term_dict[term]['logderivative2']
          for subterm in range(n_):
            expression = expression.replace(keys[iterate+subterm],values[iterate+subterm],1)
            ## Add an exception to reference the already used variables using a keyword
            subterm_var = "_VAR"+str(subterm+1)+"_"
            expression = expression.replace(subterm_var,values[iterate+subterm])
          iterate+=n_
          f.write("  ret += {}\n".format(expression.replace("_s_","S")))
          #f.write("  ret = np.add(ret,{},out=ret,casting='unsafe')\n".format(expression.replace("_s_","S")))
        f.write("  return ret\n")
        f.write("logderivative2 = frompyfunc(fp,{},1)\n".format(self.N_terms))

  def set_fingerprint(self,fp_hash):
    self.fingerprint = fp_hash

    print(fp_hash)

    ## Add a list to collect results dictionaries under this has
    if(self.fingerprint not in self.results): self.results[self.fingerprint] = []
    
    hash_list = fp_hash.split(":")
    print(hash_list)
    name = hash_list[0]
    hash_mode = hash_list[1]
    self.fingerprint_N = hash_list[2]
    if(self.fingerprint_N!="max"): self.fingerprint_N = int(self.fingerprint_N)
    self.sample_mode = hash_mode ## Important for random samples


    count = [ term_dict[term]["n"] for term in hash_list[3:]]
    self.N_terms = np.sum(count) 
 
    print(count)
    print(self.N_terms)

    ## If we have seen this fingerprint before, just load up the previous file
    if(self.fingerprint in self.fingerprint_function_dict and hash_mode == 'first'):
      print(self.function)
      print(self.fingerprint_function_dict)
      self.function = self.fingerprint_function_dict[self.fingerprint]
      print(self.function)
      #with open(".\\Functions\\{}.py".format(self.function),"r") as f: flines = f.readlines()
      if('fingerprint' in globals().keys()): del globals()['fingerprint']
      if('logderivative' in globals().keys()): del globals()['logderivative']
      if('logderivative2' in globals().keys()): del globals()['logderivative2']
      
      ## Executes the python in the custom file (i.e. a function)
      with open("./Functions/{}.py".format(self.function),"r") as f: flines = f.readlines()
      exec("".join(flines),globals())
      with open("./Functions/{}_logderivative.py".format(self.function),"r") as f: flines = f.readlines()
      exec("".join(flines),globals())
      with open("./Functions/{}_logderivative2.py".format(self.function),"r") as f: flines = f.readlines()
      exec("".join(flines),globals())

      print("Function is reset to {}!".format(self.function))
      #print("LOAD log derivative???")
      return

    ## Otherwise write a vectorized loss function in terms of the input data etc. 
    key = "".join([np.random.choice(string_list) for i in range(10)])

    ## Write the function and its log derivative
    self.write_function(key)

    ## Overwrite any existing function with the name 
    if('fingerprint' in globals().keys()): del globals()['fingerprint']
    if('logderivative' in globals().keys()): del globals()['logderivative']
    if('logderivative2' in globals().keys()): del globals()['logderivative2']
    #with open(".\\Functions\\{}.py".format(key),"r") as f: flines = f.readlines()
    with open("./Functions/{}.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative2.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())

    self.fingerprint_function_dict[self.fingerprint] = key
    self.function = key

  ## Searching for a good starting point
  def preseed(self, num_samples, logd = False):
    p0 = np.random.uniform(low=-10,high=10,size=[num_samples, self.N_terms])
    #p0 = np.random.choice([np.sqrt(2),0.5,1,2,3], size = [num_samples, self.N_terms])


    if(self.fit_mode=="log"):
      if(logd == True):
        f1 = self.real_logd_loss
        f2 = self.imag_logd_loss
      else:
        f1 = self.real_log_loss
        f2 = self.complex_log_loss
    if(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()
    losses = [self.real_log_loss(q) for q in p0]
    print(losses)
    print(np.amin(losses))
    print(np.amax(losses))
    amin = np.argmin(losses)
    print(amin)
    print(p0[amin])

    exit()



  ## A way of calling BFGS on a subspace of the parameters...
  def partial_BFGS(self, fix_dict, p0=None, order=0):

    print(fix_dict) ## Try to get the constraints in as well for params which must be >0
  
    ## Look into fix dict
    states = np.array([fix_dict[i]["fixed"] for i in fix_dict.keys()])
    #mask = ??? ## zeros ones
    ## Let values be 0
    values = np.array([fix_dict[i]["value"] for i in fix_dict.keys()])
    print(states)
    print(values)

    num_free_vars = np.sum(~states)
    print(num_free_vars)

    if( p0 == None): p0 = np.random.uniform(low = -1, high = 1, size = num_free_vars)

    ## Taking p0, states and values, expand to full vector
    print(p0)

    #full = states * values + deal(p0, states)
    #print(full) 


    if(self.fit_mode=="log"):
      f1 = self.partial_real_log_loss
      f2 = self.partial_complex_log_loss
    if(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()

    ## Figure out how to assemble the problem in the loss function?
    res = minimize(f1, x0=p0, args = (order, states, values), method = "BFGS", tol = 1e-6)
    res = minimize(f2, x0=res.x, args = (order, states, values), method = "BFGS", tol = 1e-8)
    x_final = states*values + deal(res.x,states) 
    loss = f2(res.x, order, states, values)
    self.register(x_final,loss)
    return x_final, loss
  
  ## Vectorised difference function
  def partial_real_log_loss(self,p, order, states,  values):
    full = states*values + deal(p, states)
    if(order == 0): 
      A = fingerprint(*full)[0]
      B = np.abs(np.real(A)-self.real_logm)
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*full)[0]
      B = np.abs(np.real(A)-np.real(self.ratio))
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      A = logderivative2(*full)[0]# - logderivative(*p)[0]**2
      B = np.abs(np.real(A)-np.real(self.ratio2)+np.real(self.ratio**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)
 
  ## WE CAN PROBABLY TIDY THIS UP USING DEFAULT ARGUMENTS
  ## Vectorised difference function
  def partial_complex_log_loss(self,p,order, states, values):
    full = states*values + deal(p, states)
    if(order == 0): 
      A = fingerprint(*full)
      B = np.abs(np.real(A)-self.real_logm)
      B = np.maximum(0.0,B-self.real_log_diff)
      C = np.abs(wrap(np.imag(A)-self.imag_logm))
      C = np.maximum(0.0,C-self.imag_log_diff)
    if(order == 1): 
      A = logderivative(*full)
      B = np.abs(np.real(A)-np.real(self.ratio))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio)))
    if(order == 2): 
      A = logderivative2(*full)# - logderivative(*p)**2
      B = np.abs(np.real(A)-np.real(self.ratio2)+np.real(self.ratio**2))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2)+np.imag(self.ratio**2)))
    return np.mean(B+C)

    


  ## A gradient based approach to get the optimial parameters for a given fingerprint
  def BFGS(self,p0=None, order=0):
    #self.BFGS_derivative_order = derivative_order
    #if(p0==None): p0=np.random.random(self.N_terms)
    if(p0==None): p0=np.random.uniform(low=-1,high=1,size=self.N_terms)
    if(self.fit_mode=="log"):
      f1 = self.real_log_loss
      f2 = self.complex_log_loss
    if(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()

    print("ADDING BFGS CONSTRAINTS MECHANISM")
    print("Optional callable jac, hess, etc.")

    #boundssequence or Bounds, optional
    #Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods. There are two ways to specify the bounds:
    #Instance of Bounds class.
    #Sequence of (min, max) pairs for each element in x. None is used to specify no bound.

    ## Really easy... generate in fix_dict... ? i.e. both positive?
    ## Hard if they are just equations? This would be done as a linear constraint? i.e. a + b (s_max) > 0 and a + b (s_min) > 0

    ## switch method to L-BFGS-B

    print("WARNING: CONSTRAINTS APPLIED TO ALL VARAIBLES... _poly-coeff_ may actually need to be negative?")
    constraints = [(0,None) for i in p0]

    res = minimize(f1, x0=p0, args = (order), method = "L-BFGS-B", bounds = constraints, tol = 1e-6)
    res = minimize(f2, x0=res.x, args = (order), method = "L-BFGS-B", bounds = constraints, tol = 1e-8)
    loss = f2(res.x, order)
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
    self.results["descriptors"] = descriptors
    self.results[self.fingerprint].append({"params" : params, "loss" : loss})

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
  def real_log_loss(self,p, order):
    if(order == 0): 
      A = fingerprint(*p)[0]
      B = np.abs(np.real(A)-self.real_logm)
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*p)[0]
      B = np.abs(np.real(A)-np.real(self.ratio))
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      A = logderivative2(*p)[0]# - logderivative(*p)[0]**2
      B = np.abs(np.real(A)-np.real(self.ratio2)+np.real(self.ratio**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)
  
  ## Vectorised difference function
  def complex_log_loss(self,p,order):
    if(order == 0): 
      A = fingerprint(*p)
      B = np.abs(np.real(A)-self.real_logm)
      B = np.maximum(0.0,B-self.real_log_diff)
      C = np.abs(wrap(np.imag(A)-self.imag_logm))
      C = np.maximum(0.0,C-self.imag_log_diff)
    if(order == 1): 
      A = logderivative(*p)
      B = np.abs(np.real(A)-np.real(self.ratio))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio)))
    if(order == 2): 
      A = logderivative2(*p)# - logderivative(*p)**2
      B = np.abs(np.real(A)-np.real(self.ratio2)+np.real(self.ratio**2))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2)+np.imag(self.ratio**2)))
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

  def point_evaluation(self, q, order = 0):
    return self.real_log_loss(q, order), self.complex_log_loss(q, order)

  ## A function which suggests possible closed forms
  def speculate(self, k = 4):
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

    ## A place to store lists of samples

    results_dict = {}

    #sample_dict = {i : [] for i in terms_list.keys()}
    #terms_dict = {i : [] for i in terms_list.keys()}

    ## For each parameter get the correct list:KDTree pair
    for par,value in zip(terms_list.keys(),self.best_params):
      term_type = terms_list[par]
      ## Could do a search within  tolerance or just do nearest neighbours


      ## For the parameters that were fixed to squared to keep 
      if( term_type == "_sqrtnotzero_"):
        print("*** ",par,"**2 :",term_type," ***")
        distances, IDs = global_trees_dict[term_type].query([np.abs(value**2)],k)
      else:
        print("*** ",par,":",term_type," ***")
        distances, IDs = global_trees_dict[term_type].query([np.abs(value)],k)

      dict_IDs = [ list(global_constants_dict[term_type].keys())[i] for i in IDs]

      ## Use softmax to get probabilities?
      d_sum = np.sum([ np.exp(-d/np.mean(distances)) for d in distances ])
      probs = [ np.exp(-d/np.mean(distances))/d_sum for d in distances ]

      results_dict[par] = {}
      results_dict[par]["term_type"] = term_type
      results_dict[par]["best"] = value
      results_dict[par]["distances"] = distances
      results_dict[par]["probs"] = probs
      results_dict[par]["guesses"] = dict_IDs

      #### ADD SOMETHING LIKE if _sqrtnotzero_, just do 'best', or historical values, or random...
      #### Later we might want to make a comprehensive dict of values like 2^(-7/2), but could be huge...
      #### For gamma linear just do the best few
      #if( term_type == "_sqrtnotzero_" ):
      #  sample_dict[par] = [value for i in range(samples)]        
      #else:
      #  samp = np.random.choice([i for i in range(k)], size = samples, p = probs)
      #  print(samp)
      #  terms_dict[par] = [ dict_IDs[j] for j in samp]
      #  sample_dict[par] = [global_constants_dict[term_type][q] for q in terms_dict[par]]
      #  terms_dict[par] = dict_IDs.copy()

      for i, delta, p in zip(dict_IDs, distances, probs):
        delta = np.round(delta, decimals = 8)
        p = np.round(p, decimals = 3)
        ##i = i.replace("/1","")
        if( term_type == "_sqrtnotzero_"):
          print(value**2,"~","{}".format("-" if value < 0 else "")+i," (Delta = {}, p = {})".format(delta,p))
        else:
          print(value,"~","{}".format("-" if value < 0 else "")+i," (Delta = {}, p = {})".format(delta,p))
      
      print("*** ","~~~"," ***") 

    #p0 = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
    #print("Ideal (test)--> ", p0)
    #if(samples == None): return []

    ## Need to overhaul this such that on the first pass it generates all of the information
    ## Then below it does a clean sampling on obviously named lists and writes out the results
    #print(results_dict)


    #samp_points = []
    ## Generate samples
    #print(sample_dict)
    #samples = np.vstack(list(sample_dict.values())).T
    #terms = np.vstack(list(terms_dict.values())).T
    #samples = np.unique(samples,axis = 0)
    #terms = np.unique(terms,axis = 0)
    #probs = 0
    #print(samples.shape)
    return results_dict
    #exit()

  def most_likely_from_results(self, results):
    p = []
    for i in results.keys():
      am = np.argmax(results[i]["probs"])
      #print(i,results[i],results[i]["guesses"][am])
      p.append(results[i]["guesses"][am])
    return p

  def print_function_from_guess(self, p, language = "MMA", s_value = "s"):
    print(self.fingerprint)
    fp_list = self.fingerprint.split(":")[3:]
    print(fp_list)
    string_list = []
    itr = 0
    for i in fp_list:
      #print(term_dict[i])
      #print(term_dict[i][language])
      expression = term_dict[i][language]
      for tt in range(term_dict[i]["n"]):
        expression = expression.replace(term_dict[i]["terms"][tt],p[itr],1)
        itr+=1
      string_list.append(expression)
    string = " * ".join(string_list)
    string = string.replace("_s_", s_value)
    return string

  ## Given a sympy result attempt the inverse Mellin Transform
  ## This should return a functional form that can be used to inspect the original data...
  def compute_inverse_mellin_transform(self, fingerprint, start = 0):
    from sympy import inverse_mellin_transform, oo, gamma, sqrt
    from sympy.abc import x, s
    result = eval("inverse_mellin_transform({}, s, x, ({}, oo))".format(fingerprint,start))
    return result

  def get_normalisation_coefficient(self, fingerprint):
    from sympy import gamma, sqrt
    from sympy.abc import s
    print(fingerprint)
    result = eval(fingerprint)
    print(result)
    return result


  ## A function that attempts to generally solve with a given fingerprint
  def cascade_search(self):
    ## First do a few high order BFGS to get gamma parameters
    print("Getting Order 2 BFGS")
    for i in range(10):
      self.BFGS(order = 2)
    ## Get the best params,
    ## Check for consistency in order 1 and order 0
    results = self.speculate()
    #print(results)
    p_best = self.most_likely_from_results(results)
    print(p_best)

    #####
    ## OPTIONAL --> Do the other fittings anyway just to check.. 
    #####
 
    fp_list = self.fingerprint.split(":")[3:]
    print(p_best)
    print(fp_list)
    for ii in fp_list:
      print(term_dict[ii])
    par_nums = [term_dict[ii]['n'] for ii in fp_list]
    print(par_nums)
    print(self.best_function)

    order_2_terms = ["shift-gamma","neg-shift-gamma","linear-gamma","neg-linear-gamma","scale-gamma","neg-scale-gamma","neg-P1","P1","alt-linear-gamma","alt-neg-linear-gamma"]
    order_1_terms = ["c^s"] + order_2_terms 

    positive_terms = ["c","c^s"]

    ## Figure out which parameters are order 2 and could be fixed...


    fix_dict = {}
    ## Determine which parameters are which
    itr = 0
    for ii in fp_list:
      for jj in range(term_dict[ii]['n']):
        token = "p{}".format(itr)
        fix_dict[token] = {}
        if(ii in order_2_terms):
          fix_dict[token]["fixed"] = True
          fix_dict[token]["value"] = float(eval(p_best[itr]))
        else:
          fix_dict[token]["fixed"] = False
          fix_dict[token]["value"] = 0
        itr+=1

    #print(fix_dict)
    #input()


    ## Do a (constrained) order 1?
    if( "c^s" in fp_list):
      res = []
      for i in range(10):
        ret = self.partial_BFGS(fix_dict, order = 1)
        res.append(ret)
      best_idx = np.argmin([r[1] for r in res])
      print("best local params", res[best_idx])
      
      ## DO SOME KIND OF CHECK ON THE LOSS?

      idx_where_cs = 1
      fix_dict["p{}".format(idx_where_cs)]["fixed"] = True
      fix_dict["p{}".format(idx_where_cs)]["value"] = res[best_idx][0][idx_where_cs]
   
    print(fix_dict)

    ## Do a constrained order 0 (for the constant) ?? Can also normalise?
    if( "c" in fp_list):
      res = []
      for i in range(10):
        ret = self.partial_BFGS(fix_dict, order = 0)
        res.append(ret)
      best_idx = np.argmin([r[1] for r in res])
      print("best local params", res[best_idx])
      
      
      idx_where_c = 0
      fix_dict["p{}".format(idx_where_c)]["fixed"] = True
      fix_dict["p{}".format(idx_where_c)]["value"] = res[best_idx][0][idx_where_c]

    print(fix_dict)
    p_new = [fix_dict[key]["value"] for key in fix_dict.keys()]
    final_loss = self.point_evaluation(p_new, order = 0)
    print(final_loss)
    final_loss = self.point_evaluation(p_new, order = 1)
    print(final_loss)
    final_loss = self.point_evaluation(p_new, order = 2)
    print(final_loss)

    ## Hack
    self.best_params = p_new
    results = self.speculate()
   
    print(results)
    p_best = self.most_likely_from_results(results)
    print(p_best)

    string = self.print_function_from_guess(p_best, "sympy", s_value = "1")
    print(string)
    coeff = self.get_normalisation_coefficient(string) 
    print("Normalisation Coefficient:",coeff)
    
    string = self.print_function_from_guess(p_best, "sympy")
    print("Into inv. Mellin Transform",string+"/{}".format(coeff))
    
    from sympy import nsimplify
    string_in = nsimplify(string+"/{}".format(coeff))
    print("Simplified: ",string_in)
    function_guess = self.compute_inverse_mellin_transform(string_in, start = 0) 
    print(function_guess)
    print(nsimplify(function_guess, rational = True))
    


    exit()

 
    if("c" in fp_list and "c^s" not in fp_list):
      ## Take the best parameters, fix them... evalulate at s=1 and 'normalise'
      p_best[0] = "1" ## Ignore the fitted constant which is nonsense
      string = self.print_function_from_guess(p_best, "sympy", s_value = "1")
      coeff = self.get_normalisation_coefficient(string) 
      p_best[0] = "("+str(coeff)+")**(-1)"
      string = self.print_function_from_guess(p_best, "sympy", s_value = "1")
      coeff = self.get_normalisation_coefficient(string) 
      exit()



    string = self.print_function_from_guess(p_best, "sympy")
    print(string) 

    function_guess = self.compute_inverse_mellin_transform(string)
    print(function_guess)

    ## A new function that samples results of a speculate()


    ##
    ## Check for rounded parameters high high probability

    ## Specifically for 0 shift and 1 scale parameters consider a reduction in the fingerprint definition?



    ## Reenter the rounded combinations to get losses

    ## Decide on which parameters to fix...

    ## Perform a constrained BFGS! (We might have to write a different method)...
    ## Ideally it will only optimize over a few variables and expands the solution vector as it goes..

    ## For parameters of the form :c: perform a simple normalisation!

    ## For parameters of the form :c^s: perform a log(c) trick...

    #OR syMPY## Return a guess at the closed form (in Mathematica Language)
    




## A function that helps
def gen_fpdict(parlist,N='max',mode='first',name=''):
  #hash_string = ":".join(sorted(parlist))
  hash_string = ":".join(parlist)
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









