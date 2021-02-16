import numpy as np
import scipy.special as sp
import math
import random

## Custom distribution to learn
def distribution(x,y,a,b): 
 return sp.gamma(b)*sp.gamma(a)/((1.0 + x)**a)/((1.0 + y)**b)

a_real = 5
b_real = 5

sample_cut=100.0
siz = 1000000
x=np.random.uniform(0,sample_cut,size=siz)
y=np.random.uniform(0,sample_cut,size=siz)
r=np.random.uniform(0,1,size=siz)
samples = []
for a,b,c in zip(x,y,r):
  if(c < distribution(a,b,a_real,b_real)): samples.append((a,b))

s1 = np.random.uniform(0,10,size=100)
s2 = np.random.uniform(0,10,size=100)

x,y = np.array([k[0] for k in samples]), np.array([k[1] for k in samples])
print("#={}".format(len(samples)))



best_a = 999
best_b = 999
lowest_loss = 1e40

losses = []
for kk in range(1000000):  

  a_trial = random.uniform(1.0,9.0)
  b_trial = random.uniform(1.0,9.0)
  loss = 0.0
  for i in range(len(s1)):
    logE = np.log(np.mean(x**(s1[i]-1)*y**(s2[i]-1)))
    #print("logE = {}".format(logE))  
    logM = sp.gammaln(a_trial-s1[i]) + sp.gammaln(b_trial-s2[i]) + sp.gammaln(s1[i]) + sp.gammaln(s2[i])
    #print("logM = {}".format(logM))
    loss += (logE-logM)**2
  if(loss < lowest_loss):
    lowest_loss =loss
    best_a = a_trial
    best_b = b_trial
    print(best_a)
    print(best_b)
    print(lowest_loss)
  #losses.append(loss)

  #dlda1 = 2.0*(logM-logE)*(sp.digamma(b-a1)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  #dlda2 = 2.0*(logM-logE)*(sp.digamma(b-a2)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  #dldb =  2.0*(logM-logE)*(sp.digamma(b-a1)+sp.digamma(b-a2)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  
  #dLda = sp.digamma(a_trial - s1)
  #dLdb = sp.digamma(b_trial - s2)
  #eta = 0.4
  
  #a_trial = a_trial - eta*dLda
  #b_trial = b_trial - eta*dLdb
  #print(a_trial)
  #print(b_trial)
