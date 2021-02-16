#This file will generate the moments of an exponential distribution and fit them
import numpy as np
from scipy import special as sp

def logE(a,s): return sp.gammaln(a+s)
def dlogEda(a,s): return sp.digamma(a+s)


initial_guess_a = 3.14

exponent_scale = 1

for i in range(100):
    initial_guess_a = np.random.uniform(-1.0,1.0)
    samples = np.random.exponential(scale=1.0,size=10000)
    exponents = 1.0 + np.random.exponential(scale=exponent_scale,size=1000)
    a = initial_guess_a
    learning_rate = 0.01
    dLda = 0.0
    for exp in exponents:
      s = exp-1.0
      moment = np.mean(samples**s)
      logM = np.log(moment)
      #print("Moment({}) = {}".format(s,moment))
      loss = (logE(a,s) - logM)**2
      olddLda = dLda
      dLda = 2.0*(logE(a,s)-logM)*dlogEda(a,s)
      #print("Loss = {}".format(loss))
      #print("a = {}".format(a))
      #print("dLda = {}".format(dLda))
      olda = a
      a=a-learning_rate*dLda
      da = a-olda
      dd = dLda- olddLda
    ## Fit a very simple model which is a ratio of two gamma functions

    exponents = 1.0 + np.random.exponential(scale=exponent_scale,size=10000)
    values = []
    learning_rate = 0.0001
    for exp in exponents:
      s = exp-1.0
      moment = np.mean(samples**s)
      logM = np.log(moment)
      #print("Moment({}) = {}".format(s,moment))
      loss = (logE(a,s) - logM)**2
      olddLda = dLda
      dLda = 2.0*(logE(a,s)-logM)*dlogEda(a,s)
      #print("Loss = {}".format(loss))
      #print("a = {}".format(a))
      #print("dLda = {}".format(dLda))
      values.append(a)
      a=a-learning_rate*dLda
      da = a-olda
      dd = dLda- olddLda
    print("{}".format(np.mean(values),np.std(values)))
