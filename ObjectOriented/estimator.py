from moments import MomentsBundle
import numpy as np
from scipy.optimize import minimize
from sympy import nsimplify

from typing import Any
from predefined_complexity import *
from utils import get_random_key, parse_terms, wrap, deal, gen_fpdict

from moments import ExactLearningResult




class ExactEstimator:
  """
  Exact Estimator

  This version does not load files
  Extension to load from files
  """
  def __init__(self, input : MomentsBundle):

    # Defined variables
    self.tag = input.name
    self.sample_mode = "first"
    self.fit_mode = "log"
    self.N_terms = None

    # self.BFGS_derivative_order = 0

    ## These are the object
    ## The fingerpint is the dict
    ## The function is a key that is a file (could eventually make a string) containing the function

    # Placeholder variables
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
    self.s_values = input.complex_s_samples

    ## Get the dimension of the s_values
    if(len(self.s_values.shape) == 1):
      self.n_s_dims = 1
    else:
      if(len(self.s_values.shape) > 2):
        print("Error: Tensor values moments detected! dim s = {}".format(self.s_values.shape))
        exit()
      self.n_s_dims = self.s_values.shape[-1]


    self.new_moments = input.moments
    self.new_real_error = input.real_error_in_moments
    self.new_imag_error = input.imaginary_error_in_moments

    # TODO:
    # We should import a single moments object
    # [n_dims, order_k, n_s]

    self.moments     = self.new_moments[0]
    #self.logmoments  = np.log(self.moments)

    ## Dictionaries to store dq/q
    self.ratio = {}
    self.ratio2 = {}

    ### Now that we have multi-dim data, there are many possible derivatives to have.
    for i in range(1,self.n_s_dims+1):
      self.ratio["{}".format(i)] = self.new_moments[1]/self.new_moments[0]

    ### Second order
    for i in range(1,self.n_s_dims+1):
      for j in range(1,self.n_s_dims+1):
        if(j<i): continue
        self.ratio2["{}{}".format(i,j)] = self.new_moments[2]/self.new_moments[0]

    ### Third Order?

    #print("Warning: second order not actually loaded!")
    #print("Warning: Do detect max order etc.")

    ### Load in the ratio dq/q which is expected to be a
    #  sum of digamma functions
    #self.ratio = np.load("{}/logderivative_{}.npy".format(folder,tag))
    #self.ratio2 = np.load("{}/logderivative2_{}.npy".format(folder,tag))

    real_errors_exist = input.real_errors_exist
    imag_errors_exist = input.imag_errors_exist

    #TODO: Exploit the higher order errors in derivatives

    ## If the errors are to be found
    if(real_errors_exist):
      self.real_error = input.real_error_in_moments[0]
    else:
      print("Warning, assuming zero real error on moments")
      self.real_error = np.zeros(self.moments.shape)
    
    if(imag_errors_exist):
      self.imag_error = input.imaginary_error_in_moments[0]
    else:
      print("Warning, assuming zero imaginary error on moments")
      self.imag_error = np.zeros(self.moments.shape)
    

    # TODO: Check these are even used
    # self.real_s = np.real(self.s_values)
    # self.imag_s = np.imag(self.s_values)
    # self.real_m = np.real(self.moments)
    # self.imag_m = np.imag(self.moments)
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
 


  def write_function(self, key):
    """
    Write a vectorized function to evaluate arbitary fingerprint
    The method is passed a dictionary to do this

    Function to write a 'vectorised function file' from a fingerprint
    the 'fingerprint' is a shorthand notation for an Exact Learning ansatz.
    'key' is a code that makres the function unique and referencable.
    """
    print("Writing function: ",self.fingerprint,key)
    params = self.fingerprint.split(":")[3:]

    if(self.fit_mode == "log"):
      requirements = set( term_dict[term]["logreq"] for term in params )
      terms = ":".join([ term_dict[term]["logmoment"] for term in params])
      logderivative_requirements = set( term_dict[term]["logderivativereq"] for term in params )
      logderivative_terms = ":".join([ term_dict[term]["logderivative"] for term in params])
      logderivative2_requirements = set( term_dict[term]["logderivative2req"] for term in params )
      logderivative2_terms = ":".join([ term_dict[term]["logderivative2"] for term in params])
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

  def set_fingerprint(self, fp_hash) -> None:
    """
    Ingest a fingerprint hash
    """
    self.fingerprint = fp_hash

    # Add a list to collect results dictionaries under this has
    if(self.fingerprint not in self.results): 
      self.results[self.fingerprint] = []
    
    hash_list = fp_hash.split(":")
    name = hash_list[0]
    hash_mode = hash_list[1]
    self.fingerprint_N = hash_list[2]
    if(self.fingerprint_N!="max"): self.fingerprint_N = int(self.fingerprint_N)
    self.sample_mode = hash_mode ## Important for random samples
    count = [ term_dict[term]["n"] for term in hash_list[3:]]
    self.N_terms = np.sum(count) 
 
    # If we have seen this fingerprint before, just load up the previous file
    if(self.fingerprint in self.fingerprint_function_dict and hash_mode == 'first'):
      self.function = self.fingerprint_function_dict[self.fingerprint]
      self.set_ansatz(self.function)
      print("Function is reset to {}!".format(self.function))
      return

    # Otherwise write a vectorized loss function in terms of the input data etc.
    # Write the function and its log derivative
    key = get_random_key()
    self.write_function(key)
    self.set_ansatz(key)
    self.fingerprint_function_dict[self.fingerprint] = key
    self.function = key

  def set_ansatz(self, key):
    """
    Set the fingerprint and its derivatives in the global space
    """
    if('fingerprint' in globals().keys()): del globals()['fingerprint']
    if('logderivative' in globals().keys()): del globals()['logderivative']
    if('logderivative2' in globals().keys()): del globals()['logderivative2']
    
    # Executes the python in the custom file (i.e. a function)
    # This defines a new function
    with open("./Functions/{}.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative2.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())

  def preseed(self, num_samples, logd = False):
    """
    TODO: # Searching for a good starting point
    """
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

  def partial_BFGS(self, fix_dict, p0=None, order=0):
    """
    TODO: ## A way of calling BFGS on a subspace of the parameters...
    """
    ## Look into fix dict
    states = np.array([fix_dict[i]["fixed"] for i in fix_dict.keys()])
    #mask = ??? ## zeros ones
    ## Let values be 0
    values = np.array([fix_dict[i]["value"] for i in fix_dict.keys()])
    num_free_vars = np.sum(~states)
    #TODO: Consider initialisation
    if( p0 == None): 
      p0 = np.random.uniform(low = -1, high = 1, size = num_free_vars)
    if(self.fit_mode=="log"):
      f = self.log_loss
    elif(self.fit_mode == "normal"):
      print("Normal BFGS not currently supported!")
      exit()

    ## Figure out how to assemble the problem in the loss function?
    res = minimize(f, x0=p0, args = (order, 'real', states, values), method = "BFGS", tol = 1e-6)
    res = minimize(f, x0=res.x, args = (order, 'complex', states, values), method = "BFGS", tol = 1e-8)
    x_final = states*values + deal(res.x, states) 
    loss = f(res.x, order, 'complex', states, values)
    self.register(x_final,loss)
    return x_final, loss
  
  def log_loss(self, p, order, type='complex', states=None, values=None):
    """
    All types of log loss in one function
    real, complex, full, partial
    """
    if(states is not None and values is not None):
      is_partial = True
      full = states * values + deal(p, states)
    else:
      full = p

    if(not type in ['real','complex']):
      print(type)
      breakpoint()
      raise ValueError('Type must be "real" or "complex"')

    if(order == 0):
      A = fingerprint(*full)
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
      if(type=='complex'):
        C = np.abs(wrap(np.imag(A)-np.imag(np.log(self.moments))))
        C = np.maximum(0.0,C-self.imag_log_diff)
    if(order == 1):
      A = logderivative(*full)
      B = np.abs(np.real(A)-np.real(self.ratio['1']))
      if(type=='complex'): 
        C = np.abs(wrap(np.imag(A)-np.imag(self.ratio['1'])))  
    if(order == 2):
      A = logderivative2(*full)
      B = np.abs(np.real(A)-np.real(self.ratio2['11'])+np.real(self.ratio['1']**2))
      if(type=='complex'): 
        C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2['11'])+np.imag(self.ratio['1']**2)))
      
    if(type=='real'): return np.mean(B)
    if(type=='complex'): return np.mean(B+C)
    

  def partial_real_log_loss(self, p, order, states, values):
    """
    TODO: ## Vectorised difference function
    """
    full = states * values + deal(p, states)
    if(order == 0): 
      A = fingerprint(*full)
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*full)
      B = np.abs(np.real(A)-np.real(self.ratio['1']))
      # TODO: Whatever the self.real_log_diff is doing above
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      A = logderivative2(*full)# - logderivative(*p)[0]**2
      B = np.abs(np.real(A)-np.real(self.ratio2['11'])+np.real(self.ratio['1']**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)
 
  def partial_complex_log_loss(self, p, order, states, values):
    """
    TODO:   ## WE CAN PROBABLY TIDY THIS UP USING DEFAULT ARGUMENTS
    ## Vectorised difference function
    """
    full = states*values + deal(p, states)
    if(order == 0): 
      A = fingerprint(*full)
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
      C = np.abs(wrap(np.imag(A)-np.imag(np.log(self.moments))))
      C = np.maximum(0.0,C-self.imag_log_diff)
    if(order == 1): 
      A = logderivative(*full)
      B = np.abs(np.real(A)-np.real(self.ratio['1']))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio['1'])))
    if(order == 2): 
      A = logderivative2(*full)# - logderivative(*p)**2
      B = np.abs(np.real(A)-np.real(self.ratio2['11'])+np.real(self.ratio['1']**2))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2['11'])+np.imag(self.ratio['1']**2)))
    return np.mean(B+C)

  def real_log_loss(self, p, order):
    """
    TODO: Remove duplicity
    ## Vectorised difference function
    """
    if(order == 0):
      A = fingerprint(*p)
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*p)
      B = np.abs(np.real(A)-np.real(self.ratio["1"]))
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      A = logderivative2(*p)# - logderivative(*p)**2
      B = np.abs(np.real(A)-np.real(self.ratio2["11"])+np.real(self.ratio["1"]**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)

  def complex_log_loss(self, p, order):
    """
    ## Vectorised difference function
    TODO: Remove duplicity, only one loss function
    Real/Imag, total/partial, order
    """
    if(order == 0): 
      A = fingerprint(*p)
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
      C = np.abs(wrap(np.imag(A)-np.imag(np.log(self.moments))))
      C = np.maximum(0.0,C-self.imag_log_diff)
    if(order == 1): 
      A = logderivative(*p)
      B = np.abs(np.real(A)-np.real(self.ratio["1"]))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio["1"])))
    if(order == 2): 
      A = logderivative2(*p)# - logderivative(*p)**2
      B = np.abs(np.real(A)-np.real(self.ratio2["11"])+np.real(self.ratio["1"]**2))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2["11"])+np.imag(self.ratio["1"]**2)))
    return np.mean(B+C)

  def BFGS(self, p0=None, order: int = 0, method : str = 'L-BFGS-B') -> tuple[Any, float]:
    """
    TODO: ## A gradient based approach to get the optimial parameters for a given fingerprint
    p0: A first guess of parameters
    order: Which moment derivative we will fit to, [0,1,2]
    """
    #if(method != "BFGS"):
    #  raise NotImplementedError("method needs to be BFGS!")

    #TODO: Consider better initialisation schemes
    if(p0 is None): 
      p0 = np.random.uniform(low=-1,high=1,size=self.N_terms)

    if(self.fit_mode=="log"):
      if(self.n_s_dims > 1):
        #TODO: Extend this
        raise NotImplementedError('Only 1D functions implemented at the moment.')
      f = self.log_loss

    elif(self.fit_mode == "normal"):
      #TODO: Extend this
      raise NotImplementedError("BFGS on direct moments not currently supported!")

    # TODO: print("ADDING BFGS CONSTRAINTS MECHANISM")
    # TODO: print("Optional callable jac, hess, etc.")
    #boundssequence or Bounds, optional
    #Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods. There are two ways to specify the bounds:
    #Instance of Bounds class.
    #Sequence of (min, max) pairs for each element in x. None is used to specify no bound.
    ## Really easy... generate in fix_dict... ? i.e. both positive?
    ## Hard if they are just equations? This would be done as a linear constraint? i.e. a + b (s_max) > 0 and a + b (s_min) > 0
    ## TODO: Option to switch method to L-BFGS-B

    if(method == "BFGS"):
        # Do a coarse minimisation
        res = minimize(f, x0=p0, args = (order, 'real'), method = "BFGS", tol = 1e-6)
        # Do a fine minimization
        res = minimize(f, x0=res.x, args = (order, 'complex'), method = "BFGS", tol = 1e-8)
    elif(method == 'L-BFGS-B'):
        print("WARNING: CONSTRAINTS APPLIED TO ALL VARIABLES... _poly-coeff_ may actually need to be negative!")
        print("WARNING: CONSTRAINTS APPLIED TO ALL VARIABLES... some gamma coeffs may actually need to be negative!")
        print("WARNING: Some log(x) coeffs cannot be zero!")
        # TODO: Explain this, need to expand to understand the exact fingerprint being used.
        # This is poly-coeff, which is a polynomial coefficient, and yes can be negative, so need to check the ansatz
        small = 1e-6
        constraints = [(0, None) for _ in p0]
        for i, term in enumerate(self.fingerprint.split(":")[3:]):
          #TODO: This assumes c and c^s are at the front of the list? (i.e. double param terms etc.)
          if(term in ['c','c^s']):
            constraints[i]=(small,None)
        # Do a coarse minimisation
        res = minimize(f, x0=p0, args = (order, 'real'), method = "L-BFGS-B", bounds = constraints, tol = 1e-6)
        # Do a fine minimization
        res = minimize(f, x0=res.x, args = (order, 'complex'), method = "L-BFGS-B", bounds = constraints, tol = 1e-8)

    # Calculate final (fine) loss
    loss = f(res.x, order, 'complex')

    # Register this result (in case it is the best ever)
    self.register(res.x, loss)

    print("Warning: Registering losses of different orders equivalently.")
    return res.x, loss

  def register(self, params, loss) -> None:
    """
    ## Store the results
    TODO: Best Solution (eventually have top k solutions BPQ)
    """
    
    if(loss < self.best_loss):
      print("New record solution! {}\n{}".format(loss,self.fingerprint))
      self.best_loss = loss
      self.best_params = params
      self.best_fingerprint = self.fingerprint
      self.best_function = self.function

    # TODO: Is this even used?
    descriptors = self.descriptors_from_fingerprint(self.fingerprint)
    self.results["descriptors"] = descriptors
    self.results[self.fingerprint].append({"params" : params, "loss" : loss})

  def descriptors_from_fingerprint(self,fingerprint):
    """
    TODO:
    ## Convert params to a unifed set of descriptors
    ## This may be useful to train models that predict loss via function composition
    ## Consider a Free-Wilson type analysis
    """
    fp_list = fingerprint.split(":")[3:]
    descriptors = []
    elements = ["c","c^s","linear-gamma","scale-gamma","shift-gamma"]
    descriptors = [ fp_list.count(e) for e in elements ]
    return descriptors

  def set_mode(self, mode: str) -> None:
    """
    TODO: What is this?
    Set the fitting mode of the estimator
    """
    if(mode not in ["log", "normal"]):
      raise NotImplementedError("Error: valid mode choices are 'log' or 'normal' for logmoments or moments respectively!")
    self.fit_mode = mode

  def summarise(self) -> None:
    """
    TODO: Describe this
    Run through the results we have, get the minimum loss registered
    Print out the best params
    """
    results = self.results
    keys = self.results.keys()
    losses = [[j["loss"] for j in results[key]] for key in keys]
    minimum_loss = np.amin(losses,axis=1)
    for i,j in zip(keys, minimum_loss):
      print(i,j)

  def point_evaluation(self, q, order = 0):
    """
    Evalulate the total loss at a point and order
    TODO: Generalised loss mode.
    """
    return self.real_log_loss(q, order), self.complex_log_loss(q, order)

  def speculate(self, params, k = 4, silent = True):
    """
    A function that looks at params and tries to match closed forms to parameters.

    k: The number of parameter values to consider, i.e. k closest neighbours
    returns: results dict
    """
    terms_list = self.function_terms_dict[self.function]
    results_dict = {}

    ## For each parameter get the correct list:KDTree pair
    for par, value in zip(terms_list.keys(), params):
      term_type = terms_list[par]
      ## Could do a search within  tolerance or just do nearest neighbours

      ## For the parameters that were fixed to squared to keep 
      if( term_type == "_sqrtnotzero_"):
        if(not silent): print("*** ",par,"**2 :",term_type," ***")
        distances, IDs = global_trees_dict[term_type].query([np.abs(value**2)],k)
      else:
        if(not silent): print("*** ",par,":",term_type," ***")
        distances, IDs = global_trees_dict[term_type].query([np.abs(value)],k)

      dict_IDs = [ list(global_constants_dict[term_type].keys())[i] for i in IDs]

      ## Use softmax to get probabilities
      d_sum = np.sum([ np.exp(-d/np.mean(distances)) for d in distances ])
      probs = [ np.exp(-d/np.mean(distances))/d_sum for d in distances ]

      results_dict[par] = {}
      results_dict[par]["term_type"] = term_type
      results_dict[par]["best"] = value
      results_dict[par]["distances"] = distances
      results_dict[par]["probs"] = probs
      results_dict[par]["guesses"] = dict_IDs

      if(not silent):
        for i, delta, p in zip(dict_IDs, distances, probs):
            delta = np.round(delta, decimals = 8)
            p = np.round(p, decimals = 3)
            ##i = i.replace("/1","")
            if( term_type == "_sqrtnotzero_"):
              print(value**2,"~","{}".format("-" if value < 0 else "")+i," (Delta = {}, p = {})".format(delta,p))
            else:
              print(value,"~","{}".format("-" if value < 0 else "")+i," (Delta = {}, p = {})".format(delta,p))
        
        print("*** ","~~~"," ***") 
    return results_dict


  def most_likely_from_results(self, results : dict) -> dict:
    """
    Takes a results dictionary, and returns the parameter with argmax probability
    """
    p = {}
    for i in results.keys():
      am = np.argmax(results[i]["probs"])
      p[i] = results[i]["guesses"][am]
    return p

  def print_function_from_guess(self, p : dict, language : str = "MMA", s_value : str = "s") -> str:
    """
    Takes a best guess parameter vector
    Uses the implied fingerprint definition loaded in the object
    p:  The best parameter dictionary
    language:
    s_value:
    Returns: A string [in language]
    """
    
    fp_list = self.fingerprint.split(":")[3:]
    string_list = []
    itr = 0
    for i in fp_list:
      expression = term_dict[i][language]
      for tt in range(term_dict[i]["n"]):
        # TODO: Does this only replace 1? Is that OK? What if it has duplicates of the same term?
        expression = expression.replace(term_dict[i]["terms"][tt],p[f'p{itr}'],1)
        itr += 1
      string_list.append(expression)
    string = " * ".join(string_list)
    string = string.replace("_s_", s_value)
    return string

  def compute_inverse_mellin_transform(self, fingerprint, start = 0):
    """
    TODO:
    ## Given a sympy result attempt the inverse Mellin Transform
    ## This should return a functional form that can be used to inspect the original data...
    """
    from sympy import inverse_mellin_transform, oo, gamma, sqrt
    from sympy.abc import x, s
    result = eval("inverse_mellin_transform({}, s, x, ({}, oo))".format(fingerprint,start))
    return result

  def get_normalisation_coefficient(self, fingerprint):
    """
    This assumes the value of s was replaced with 1.
    Evalulates the sympy expression for the best matched fingerprint
    In theory this will find the normalisation of distributions
    """
    from sympy import gamma, sqrt
    from sympy.abc import s

    # TODO: This assumes only gamma and sqrt are name functions that might appear in the string...
    result = eval(fingerprint)
    return result

  def cascade_search(self, n_itrs=1, k=4) -> ExactLearningResult:
    """
    TODO: A function that attempts to generally solve with a given fingerprint
    """

    # terms like 'c', 'c^s', ...
    fp_list = self.fingerprint.split(":")[3:]

    if(fp_list in [['c'],['c','c^s']]):
      raise ValueError(f'cascade search not suitable for ansatz {fp_list}')

    ## First do a few high order BFGS to get gamma parameters
    print("Getting Order 2 BFGS")
    res = []
    for _ in range(n_itrs):
      # TODO: Consider refactoring BFGS to give argmin of losses?
      ret = self.BFGS(order = 2)
      res.append(ret)

    # TODO: Also argmax type approach
    best_idx = np.argmin([r[1] for r in res])
    best_local_params = res[best_idx][0]
    results = self.speculate(best_local_params, k=k)

    # Speculate top k = 4
    # TODO: We should actually try every one of the top $k$ symbolically!

    # Get a dictionary with keys pi : best_value_i
    p_best = self.most_likely_from_results(results)

    # TODO: self.likely_candidates_from_results
    # TODO: Weight the probabilities by 'complexity' i.e. 1/2 is more common than 11/20

    #####
    #TODO: OPTIONAL --> Do the other fittings anyway just to check.. 
    #####
 
    # Get the number of parameters per term (not used)
    par_nums = [term_dict[ii]['n'] for ii in fp_list]

    # TODO: Got to figure out how to handle these?
    order_2_terms = ["shift-gamma","neg-shift-gamma","linear-gamma","neg-linear-gamma","scale-gamma","neg-scale-gamma","neg-P1","P1","alt-linear-gamma","alt-neg-linear-gamma"]
    order_1_terms = ["c^s"] + order_2_terms 
    order_0_terms = []
    positive_terms = ["c","c^s"]

    # Figure out which parameters are order 2 and could be fixed...
    # TODO: Explain this more clearly?

    idx_where_cs = None
    idx_where_c = None

    fix_dict = {}
    ## Determine which parameters are which
    itr = 0
    for ii in fp_list:
      for _ in range(term_dict[ii]['n']):
        token = "p{}".format(itr)
        if(ii == 'c^s'):
          idx_where_cs = itr
        if(ii == 'c'):
          idx_where_c = itr
        fix_dict[token] = {}
        if(ii in order_2_terms):
          print("fp_list",fp_list)
          print("ii",ii)
          print("itr",itr)
          print(token)
          print(fix_dict)
          print(p_best)
          fix_dict[token]["fixed"] = True
          # TODO: This is an argmax only approach, and we might want to try all of them
          fix_dict[token]["value"] = float(eval(p_best[token]))
        else:
          fix_dict[token]["fixed"] = False
          # TODO: check this dumb value is not actually used?
          fix_dict[token]["value"] = 0
        itr+=1

    

    # Do a constrained order 1
    if( "c^s" in fp_list):
      res = []
      # TODO: break out seperate n_itrs params if needed
      for _ in range(n_itrs):
        ret = self.partial_BFGS(fix_dict, order = 1)
        res.append(ret)

      # TODO: Also argmax type approach
      best_idx = np.argmin([r[1] for r in res])
      print("best local params", res[best_idx])
      
      # TODO: DO SOME KIND OF CHECK ON THE LOSS?
      # Is it exact? Or below a core threshold?

      fix_dict["p{}".format(idx_where_cs)]["fixed"] = True
      fix_dict["p{}".format(idx_where_cs)]["value"] = res[best_idx][0][idx_where_cs]
   


    # Do a constrained order 0 (for the constant), can also normalise.
    if( "c" in fp_list):
      res = []
      for _ in range(n_itrs):
        ret = self.partial_BFGS(fix_dict, order = 0)
        res.append(ret)
      best_idx = np.argmin([r[1] for r in res])
      print("best local params", res[best_idx])
      
      # TODO: DO SOME KIND OF CHECK ON THE LOSS?
      # Is it exact? Or below a core threshold?
      
      fix_dict["p{}".format(idx_where_c)]["fixed"] = True
      fix_dict["p{}".format(idx_where_c)]["value"] = res[best_idx][0][idx_where_c]



    # These are raw parameters, straight out of BFGS
    p_new = [fix_dict[key]["value"] for key in fix_dict.keys()]


    results = self.speculate(p_new, k=4)
    p_best = self.most_likely_from_results(results)

    # TODO: Implement some kind of check based on below
    # Or return them in the dictionary.
    losses = {}
    losses['0'] = self.point_evaluation(p_new, order = 0)
    losses['1'] = self.point_evaluation(p_new, order = 1)
    losses['2'] = self.point_evaluation(p_new, order = 2)

    # By evalulating at s=1, we will get normalisation via Mellin Transform
    # TODO: Check about the whole p**2 vs p and whether that makes sense? (Seems to)
    normalisation_constant_string = self.print_function_from_guess(p_best, "sympy", s_value = "1")
    coeff = self.get_normalisation_coefficient(normalisation_constant_string)

    # TODO: Add checks for infinite and non-sensical constants
    # TODO: possibly hook in a pre-defined 'is this a distribution' kind of flag
    if(coeff == 0.0):
        raise ValueError('Cannot normalise with 0')

    print("Normalisation Coefficient:", coeff)

    # This time, it will leave it as a function of s
    string = self.print_function_from_guess(p_best, "sympy")
    string_in = string+"/{}".format(coeff)
    print("Into inv. Mellin Transform",string_in)


    string_in = nsimplify(string_in)
    print("Simplified: ",string_in)


    function_guess = self.compute_inverse_mellin_transform(string_in, start = 0) 
    print(function_guess)
   
    equation =  nsimplify(function_guess, rational = True)
    print(equation)

    result_dict = {}
    result_dict["equation"] = str(equation)
    result_dict["complex_moments"] = str(string_in)
    result_dict["num_dims"] = 1
    result_dict["losses"] = losses
    
    # Encode the mathematical result in a nice compact form
    result = ExactLearningResult(result_dict)

    return result

 
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
    ## Check for rounded parameters high high probability
    ## TODO: Specifically for 0 shift and 1 scale parameters consider a reduction in the fingerprint definition?
    ## Reenter the rounded combinations to get losses
    ## Decide on which parameters to fix...
    ## Perform a constrained BFGS! (We might have to write a different method)...
    ## Ideally it will only optimize over a few variables and expands the solution vector as it goes..
    ## For parameters of the form :c: perform a simple normalisation!
    ## For parameters of the form :c^s: perform a log(c) trick...ef   

  def standard_solve(self) -> list[ExactLearningResult]:
    """
    A method to try a standard set of fingerprints in a systematic way.

    Load a fingerprint, fit it and get a bunch of function suggestions.
    """
    print("Hello")
    # Upgrade to have trial params as well
    standard_fingerprints = [
      gen_fpdict(['c','shift-gamma']),
      gen_fpdict(['c','c^s','shift-gamma']),
      #gen_fpdict(['c','c^s','linear-gamma']),
      #gen_fpdict(['c','c^s','linear-gamma','linear-gamma']),
      #gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma'])
    ]

    #TODO: define standard seach params, lengths and times, heuristics
    results_list = []
    for std_fp in standard_fingerprints:
      self.set_fingerprint(std_fp)
      results = self.cascade_search(n_itrs = 10)
      results_list.append(results)

      # Do a check to see if it is exact already?
    print(results_list)

    for res in results_list:
      print(res, res.loss)

    # Consider a meta-dynamics type approach as well... 
    # Reason about which results are best.
    # Do go back, and compare against the interpolating function?
    # I.e. measure probability of success / confidence,
    # If we match everywhere, then it is perfect p = 1.0

    return results_list