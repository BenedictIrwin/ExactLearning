from moments import MomentsBundle
import numpy as np
from scipy.optimize import minimize

from predefined_complexity import *
from utils import get_random_key, parse_terms, wrap, deal

from moments import ExactLearningResult



class ExactEstimator:
  """
  Exact Estimator

  This version does not load files
  Extension to load from files
  """
  def __init__(self, input : MomentsBundle):

    print(input.__dict__.keys())

    self.tag = input.name

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
 


  def write_function(self,key):
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
    key = get_random_key()

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
  def partial_real_log_loss(self, p, order, states,  values):
    full = states*values + deal(p, states)
    if(order == 0): 
      A = fingerprint(*full)[0]
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*full)[0]
      B = np.abs(np.real(A)-np.real(self.ratio['1']))
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      A = logderivative2(*full)[0]# - logderivative(*p)[0]**2
      B = np.abs(np.real(A)-np.real(self.ratio2['1'])+np.real(self.ratio['1']**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)
 
  ## WE CAN PROBABLY TIDY THIS UP USING DEFAULT ARGUMENTS
  ## Vectorised difference function
  def partial_complex_log_loss(self, p, order, states, values):
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
      B = np.abs(np.real(A)-np.real(self.ratio2)+np.real(self.ratio**2))
      C = np.abs(wrap(np.imag(A)-np.imag(self.ratio2)+np.imag(self.ratio**2)))
    return np.mean(B+C)

    


  ## A gradient based approach to get the optimial parameters for a given fingerprint
  def BFGS(self,p0=None, order=0):
    #self.BFGS_derivative_order = derivative_order
    #if(p0==None): p0=np.random.random(self.N_terms)
    if(p0==None): p0=np.random.uniform(low=-1,high=1,size=self.N_terms)
    if(self.fit_mode=="log"):
      if(self.n_s_dims > 1):
        print("ND In Progress!")
        exit()
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
      B = np.abs(np.real(A)-np.real(np.log(self.moments)))
      B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 1): 
      A = logderivative(*p)[0]
      B = np.abs(np.real(A)-np.real(self.ratio["1"]))
      #B = np.maximum(0.0,B-self.real_log_diff)
    if(order == 2): 
      print('breakpoint')
      breakpoint()
      A = logderivative2(*p)[0]# - logderivative(*p)[0]**2
      B = np.abs(np.real(A)-np.real(self.ratio2["11"])+np.real(self.ratio["1"]**2))
      #B = np.maximum(0.0,B-self.real_log_diff)
    return np.mean(B)
  
  ## Vectorised difference function
  def complex_log_loss(self,p,order):
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
   

    equation =  nsimplify(function_guess, rational = True)
    print(equation)

    result_dict = {}
    result_dict["equation"] = str(equation)
    result_dict["complex_moments"] = str(string_in)
    result_dict["num_dims"] = 1
    
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
    