import numpy as np
import scipy.special as sp
import math

def distribution(x,y): 
 return np.exp(-x*y/(x+y))/(x+y)**2/43.829020301246054

sample_cut=5.0
siz = 100000

a1r=-0.3
a2r=-0.25

a1=0.1
a2=0.2
b=0.0

A1= []
A2 = []
B = []

losses = []
for kk in range(1000):  
  
  x=np.random.uniform(0,sample_cut,size=siz)
  y=np.random.uniform(0,sample_cut,size=siz)
  r=np.random.uniform(0,1,size=siz)
  samples = []
  for a,b,c in zip(x,y,r):
      if(c < distribution(a,b)): samples.append((a,b))
  
  logE = np.log(np.mean(x**(a1r-1)*y**(a2r-1)))
  #print("logE = {}".format(logE))
  logM = sp.gammaln(b-a1)+sp.gammaln(b-a2)+sp.gammaln(a1+a2-b)-sp.gammaln(2*b-a1-a2)
  #print("logM = {}".format(logM))
  loss = (logE-logM)**2
  losses.append(loss)

  dlda1 = 2.0*(logM-logE)*(sp.digamma(b-a1)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  dlda2 = 2.0*(logM-logE)*(sp.digamma(b-a2)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  dldb =  2.0*(logM-logE)*(sp.digamma(b-a1)+sp.digamma(b-a2)+sp.digamma(a1+a2-b)-sp.digamma(2*b-a1-a2))
  eta = 0.0001
  
  a1 = a1 - eta*dlda1
  a2 = a2 - eta*dlda2
  b = b - eta*dldb
  
  print("a1={}".format(a1))
  print("a2={}".format(a2))
  print("b={}".format(b))
  A1.append(a1)
  A2.append(a2)
  B.append(b)

print("a1 avg = {}".format(np.mean(A1)))
print("a2 avg = {}".format(np.mean(A2)))
print("b avg = {}".format(np.mean(B)))




