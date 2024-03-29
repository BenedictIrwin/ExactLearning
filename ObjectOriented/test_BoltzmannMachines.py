from logging import root
import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt
from ObjectOriented import *
import os
from moments import MomentsBundle
from estimator import ExactEstimator

import h5py
from matplotlib import pyplot as plt
h5 = h5py.File('TestData.h5','r')

# from exact_learning.data import points_to_moments

for i in range(50,100):
    x = h5['Dataset1'][i][:,0] 
    y = h5['Dataset1'][i][:,1]
    plt.plot(x,y,label = f'{i}')
    plt.plot([min(x),max(x)],[0,0],'k:')
    plt.legend()
    plt.show()

    # Extract moments
    mb = MomentsBundle(f"curve_{i}")
    mb.ingest(x, y)

    # Define a potential solution
    ee = ExactEstimator(mb)
    # Need to add a way to scan multiple solutions and test one by one...
    result = ee.standard_solve()

    exit()

    input("Press Enter")

exit()

names = []
roots = []
with open("test_1d_distributions.txt","r") as f:
  header = f.readline()
  for idx, line in enumerate(f):
    name, min, max, python, mma, moments = line.split(",")
    min = float(min)
    max = float(max)
    print(name, min, max, python, mma, moments)
    python = python.replace("[comma]",",") # multi-arg functions
    
    x = np.linspace(min,max,300)
    f = eval("lambda x :"+ python)
    #plt.title(name)
    #plt.xlabel("x")
    #plt.ylabel("p(x)")
    #plt.plot(x,f(x))
    #plt.show()

    # Now call Exact Learning

    # Define a problem
    mb = MomentsBundle(name)
    mb.ingest(x,f(x))

    print("Biggest realisation is for Gamma(a s + b), but a in [1,-1,1/2,-1/2]")
    print("Also b in [k/2,] from [-1/2, ... ]")
    breakpoint()

    s = np.linspace(mb.moments_sample_s_min,
                    mb.moments_sample_s_max,
                    mb.moments_sample_n_samples)
    m = np.clip(mb.moments_interpolating_function[1](s), -25,25)

    names.append(name)
    roots.append(mb.large_k1_root)

    plt.plot(s,np.log(m),label = name)
    if(mb.large_k1_root is not None):
      # Plot the dip of ther largest root
      plt.plot(mb.large_k1_root,np.log(mb.moments_interpolating_function[1](mb.large_k1_root)),'ro') 
    # For now just plot the moments have a good look for all of the problems
    if(idx == 6):
      print("Largest Roots: ")
      for i,j in zip(names, roots):
        print(i,j)
      plt.legend()
      plt.show()
      exit()
    continue

    # Define a potential solution
    ee = ExactEstimator(mb)

    # Need to add a way to scan multiple solutions and test one by one...
    result = ee.standard_solve()

    x = np.linspace(min,max,1000)
    f = eval("lambda x :"+ python)
    from numpy import exp as exp
    from numpy import sqrt as sqrt
    
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.plot(x,f(x),label='original')
    for res in result:
      g = eval("lambda x :"+ str(res.equation))
      plt.plot(x,g(x),'r:',label=f'pred {res.equation}')
      plt.legend()
    plt.show()
    input()
    

    

    # OR
    #ee = ExactEstimator()
    #ee.set_fingerprint()
    #ee.fit(mb)



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
