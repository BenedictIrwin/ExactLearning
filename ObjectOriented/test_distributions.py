import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt
from ObjectOriented import *
import os
from moments import MomentsBundle
from estimator import ExactEstimator

# from exact_learning.data import points_to_moments

with open("test_1d_distributions.txt","r") as f:
  header = f.readline()
  for line in f:
    name, min, max, python, mma, moments = line.split(",")
    min = float(min)
    max = float(max)
    print(name, min, max, python, mma, moments)
    python = python.replace("[comma]",",") # multi-arg functions
    x = np.linspace(min,max,100)
    f = eval("lambda x :"+ python)
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.plot(x,f(x))
    plt.show()

    # Now call Exact Learning

    # Define a problem
    mb = MomentsBundle(name)
    mb.ingest(x,f(x))

    # Define a potential solution
    ee = ExactEstimator(mb)

    exit()

    # OR
    ee = ExactEstimator()
    ee.set_fingerprint()
    ee.fit(mb)



    # Pass mb to an ExactEstimator Object 
    # Do this for a clone, so we can improve it without disturbing the existing one


    exit()

    # If it is not clear how, make a data converter
    # Any time we call plt.plot(x,y) we should be able to call
    
    # unknown_function = el.ingest(x,y,[options about s])

    # unknown_function -> <el.moments_bundle> 


    # Config = interval? num dims? Allowable solutions?
    # EE = ExactEstimator([<config_dict>])
    # EE.configure(<config_dict>)

    # results = EE.fit(unknown_function)

    # results ->  <el.results>
    # results.solved() -> True/False
    # results.TeXForm() -> "\exp(-x)"  # by hooking whatever sympy has


    #### Working Example... ####
   
    # Make a folder called 'name'
    os.mkdir(name)

    # Interpolate the datapoints we have as input 'i.e. function'

    # Integrate to get the momenets from the data

    # Save files into the folder

 
    EE = ExactEstimator(name, folder = name)
    EE.set_fingerprint( gen_fpdict(['c','linear-gamma']))
    n_bfgs = 1
    for i in range(n_bfgs):
      EE.BFGS(order=2)
      print("{}%".format(100*(i+1)/n_bfgs),flush=True)
    EE.speculate(k = 4)
    ############################
    
    EE.cascade_search()


#vec_int = np.vectorize(special_int)

keyword = "Beta_Distribution"

size = 1000000

a = 3
b = 4

lengths = np.random.beta(a,b,size=size)

## Show a histogram
plt.hist(lengths, bins = 100, density = True)

from scipy.special import beta

xx = np.linspace(0,1,100)
ff = xx**(a-1)*(1-xx)**(b-1)/beta(a,b)
plt.plot(xx,ff,'k-')

plt.show()


## Generate reail and imaginary part complex moments
s_size = 5
s1 = np.random.uniform(low = 1, high = 7, size = s_size)
s2 = np.random.uniform(low = -1*np.pi, high = 1*np.pi, size = s_size)
s = np.expand_dims(s1 + s2*1j, axis = 0)

## For each moment, get the expectation of s-1 for Mellin Transform
## Also get the expectation of E[logX**k X^(s-1)] which is like the kth derivative of the fingerprint.
t = np.power(np.expand_dims(lengths,1),s-1)
q = np.mean(t,axis=0)
dq = np.mean( np.expand_dims(np.log(lengths), axis = 1) * t, axis=0)
ddq = np.mean( np.expand_dims(np.log(lengths)**2, axis = 1) * t, axis=0)
dddq = np.mean( np.expand_dims(np.log(lengths)**3, axis = 1) * t, axis=0)

s = s[0]

## Save out exact learning data (complex moments)
np.save("s_values_{}".format(keyword),s)
np.save("moments_{}".format(keyword),q)
np.save("logmoments_{}".format(keyword),np.log(q))
np.save("derivative_{}".format(keyword),dq)
np.save("logderivative_1_{}".format(keyword),dq/q)
np.save("logderivative_11_{}".format(keyword),ddq/q)
np.save("logderivative_111_{}".format(keyword),dddq/q)
