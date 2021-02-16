## Generalised Training Procedure

import numpy as np
import scipy.special as sp
import random

num_data = 10000
num_dim = 1
num_exp = 100
data = np.random.exponential(scale=1.0,size=(num_data,num_dim))
#exponents = np.random.uniform(0,3,size=(num_exp,num_dim))
exponents = np.random.exponential(scale=1.0,size=(num_exp,num_dim))

## Parameters
num_gammas = 1 # For example

## There will be num_dim scale parameters
## These are actual constants with exponents which are a function of s eta^(-s)
scale_parameters = []

## There will be one s coefficient vector and a constant for each gamma
constants = []
vectors = []

minus_sign_weights = []
## External parameters
## Extrenal parameter exponents


## Problems will come from equivalence through cancellation (in models with too many terms)
## Problems will come from equivalence of terms (these may be sorted by ordering variables that are equivalent before sotrgin the observations).

best_loss = 1e40

for kappa in range(1000):
  a = sorted([random.choice([-1,0,1]) for i in range(num_dim)])
  w = np.array([[ np.random.uniform(-1.0,1.0) for i in range(num_dim)] for j in range(num_gammas)])
  alpha = [ np.random.uniform(-1.0,1.0) for i in range(num_gammas)]
  eta = [ np.random.uniform(0.0,2.0) for i in range(num_dim) ]
  beta = [ np.random.uniform(-3,3) for i in range(num_dim) ]
  q = np.array([ [ np.random.uniform(-3,3) for i in range(num_dim)] for j in range(num_dim)])
  
  loss = 0.0
  for s in exponents:
    logE = np.log(np.mean(data**(s-1)))
    logM = sum([(np.dot(q[l],s) + beta[l])*eta[l] for l in range(num_dim)]) + sum([a[m]*sp.gammaln(np.dot(w[m],s)+alpha[m]) for m in range(num_gammas)])
    loss += (logE - logM)**2
  if(loss < best_loss):
    best_loss = loss
    best_a = a
    best_w = w
    best_alpha = alpha
    best_eta = eta
    best_beta = beta
    best_q = q
    print("Best Loss = {}".format(best_loss))
    print("Best a = {}".format(best_a))
    print("Best w = {}".format(best_w))
    print("Best alpha = {}".format(best_alpha))
    print("Best beta = {}".format(best_beta))
    print("Best eta = {}".format(best_eta))
    print("Best q = {}".format(best_q))

## Run Descent from here
a=best_a
w=best_w
alpha=best_alpha
eta=best_eta
beta=best_beta
q=best_q

lr = 0.00001

for kappa in range(10000):

  ## Get the loss
  loss = 0.0
  for s in exponents:
    logE = np.log(np.mean(data**(s-1)))
    logM = sum([(np.dot(q[l],s) + beta[l])*eta[l] for l in range(num_dim)]) + sum([a[m]*sp.gammaln(np.dot(w[m],s)+alpha[m]) for m in range(num_gammas)])
    loss = (logE - logM)**2
    ## Get the gradients
    dLdeta = [lr*2.0*(logM - logE)* (np.dot(q[l],s)+beta[l])/eta[l] for l in range(num_dim)]
    dLdq = lr*2.0*(logM - logE)*np.array([[ s[j]*np.log(eta[i]) for j in range(num_dim)] for i in range(num_dim)])
    dLdbeta = [ lr*2.0*(logM - logE)*np.log(eta[l]) for l in range(num_dim)]
    dLda = [ lr*2.0*(logM - logE)*sp.gammaln(np.dot(w[i],s)+alpha[i]) for i in range(num_dim)]
    dLdw = lr*2.0*(logM - logE)*np.array([ [ a[i]*s[j]*sp.digamma(np.dot(w[i],s)+alpha[i]) for j in range(num_dim)] for i in range(num_gammas)])
    dLdalpha = [ lr*2.0*(logM - logE)*a[i]*sp.digamma(np.dot(w[i],s)+alpha[i]) for i in range(num_gammas)]           
    ## Update the variables
    eta = [eta[i] - dLdeta[i] for i in range(num_dim)]
    q -= dLdq
    beta = [beta[i] - dLdbeta[i] for i in range(num_dim)]
    a = [a[i] - dLda[i] for i in range(num_dim)]
    w -= dLdw
    alpha = [alpha[i] - dLdalpha[i] for i in range(num_gammas)]

  print("a = {}".format(a))
  print("w = {}".format(w))
  print("alpha = {}".format(alpha))
  print("beta = {}".format(beta))
  print("eta = {}".format(eta))
  print("q = {}".format(q))




