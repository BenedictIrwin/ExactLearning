import numpy as np
from scipy.spatial import KDTree

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
del not_zero_dict['0/1'] # delete zero from here, or it will crash log(x)!

## Sqrt of these constants for positive only BFGS varaibles
sqrt_not_zero_dict = { "sqrt({})".format(i) : np.sqrt(not_zero_dict[i]) for i in not_zero_dict.keys()}

## Hypergeometric arguments, should include 2**2/3**3 type numbers as well
hyp_arg_dict = rationals_dict.copy()

## For polynomial coefficients (usually rationals)
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