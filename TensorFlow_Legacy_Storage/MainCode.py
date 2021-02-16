#This file will generate the moments of an exponential distribution and fit them
import numpy as np
from scipy import special as sp
import itertools
import sys


def logfMultiHypergeometricpFq(n,p,q,a,b,A,kstar):
  return_value = 0.0
  for l in range(n):
    #print("alpha_l . k* = {}".format(np.dot(A[l],kstar)))
    dot_term = np.dot(A[l],kstar) 
    T1 = np.sum(sp.gammaln(b[l]))
    T2 = np.sum(sp.gammaln(a[l]))
    T3 = np.sum(sp.gammaln(a[l]+dot_term))
    T4 = np.sum(sp.gammaln(b[l]+dot_term))
    #print("T = ({} {} {} {})".format(T1,T2,T3,T4))
    return_value += T1 - T2 + T3 - T4
  return return_value

def logPiGeneral(n,kstar):
  return  np.sum( sp.gammaln(-kstar) )

argc = len(sys.argv)

num_dim = 3  ## The number of dimensions of the predicited probability distribution (inputs + outputs)
num_exp = 10
num_p = 2 ## The order of the top set of terms in the hypergeometric funtion
num_q = 1 ## The order of the bottom set of terms in the hypergeometric function

x0 = np.random.exponential(scale=1,size=100)
x1 = np.random.exponential(scale=1,size=100)
x2 = np.random.exponential(scale=1,size=100)

print(x0)

#Set $\mathbf{A},\mathbf{a},\mathbf{b},\mathcal{S}$ to initial values
A = np.random.uniform(-1,1,(num_dim,num_dim))
a = np.random.uniform(0,2,(num_dim,num_p))
b = np.random.uniform(0,2,(num_dim,num_q))
S = np.random.uniform(1,3,(num_exp,num_dim))

print("A = {}".format(A))
print("a = {}".format(a))
print("b = {}".format(b))
print("S = {}".format(S))

#Calculate $\mathbf{A}^{-1}$
Ainv = np.linalg.inv(A)
#Adet = np.linalg.det(A)
Adetsign, Alogabsdet = np.linalg.slogdet( A )
print("A^-1 = {}".format(Ainv))
#print("det(A) = {}".format(Adet))
#print("|det(A)| = {}".format(Adetsign*Adet))
print("log |det(A)| = {}".format(Alogabsdet))

Loss = 0.0

#For $s \in \mathcal{S}$
for s in S:
  print("s = {}".format(s))
  
  #Calculate $\log \mathcal{E}(\mathbf{s})$
  if(num_dim == 1): logE = np.log(np.mean( [ xx0**s[0] for xx0 in x0 ] ))
  if(num_dim == 3): logE = np.log(np.mean( [ xx0**s[0]*xx1**s[1]*xx2**s[2] for xx0,xx1,xx2 in zip(x0,x1,x2) ] ))
  #print("log E[x0^{}x1^{}x2^{}] ~ {}".format(s[0],s[1],s[2],logE))
  print("log E = {}".format(logE))
  
  #Calculate $\mathbf{k}^*(\mathbf{s},\mathbf{A})$ using $\mathbf{A}^{-1}$ and $\mathbf{s}$
  kstar = np.matmul(Ainv,-s)
  print("k* = {}".format(kstar))
  
  #Calculate $\log f(\mathbf{k}^*)$
  logf = logfMultiHypergeometricpFq(num_dim,num_p,num_q,a,b,A,kstar) 
  print("logf = {}".format(logf))
  
  #Calculate $\log \Pi(\mathbf{k}^*)$
  logPi = logPiGeneral(num_dim,kstar)
  print("logPi = {}".format(logPi))
  
  #Calculate $\log \mathcal{M}$ using above two terms and log(|det A|)
  logM = logf + logPi - Alogabsdet
  print("logM = {}".format(logM))

  #Add $(\log \mathcal{E} - \log\mathcal{M})^2$ to $\mathcal{L}$
  local_loss = (logE - logM)**2
  Loss += local_loss
  print("Local_loss = {}".format(local_loss))

  #Calculate $\mathrm{vec}( \alpha_l \cdot \mathbf{k}^*)= \mathbf{A}\mathbf{k}^*$
  alpha_dot_vector = np.matmul(A,kstar)
  print("alpha-dot_vector = {}".format(alpha_dot_vector))
  
  #Calculate $\mathrm{mat}(\psi(a_{lm}))$
  psi_a_matrix = sp.digamma(a)
  print("psi_a_matrix = {}".format(psi_a_matrix))
  
  #Calculate $\mathrm{mat}(\psi(b_{lm}))$
  psi_b_matrix = sp.digamma(b)
  print("psi_b_matrix = {}".format(psi_b_matrix))
 
  #Calculate $\mathrm{mat}(\psi(a_{lm}) + \alpha_l \cdot \mathbf{k}^*)$
  psi_a_dot_matrix = sp.digamma(a.transpose() + alpha_dot_vector).transpose()
  print("psi_a_dot_matrix = {}".format(psi_a_dot_matrix))
 
  #Calculate $\mathrm{mat}(\psi(b_{lm}) + \alpha_l \cdot \mathbf{k}^*)$
  psi_b_dot_matrix = sp.digamma(b.transpose() + alpha_dot_vector).transpose()
  print("psi_b_dot_matrix = {}".format(psi_b_dot_matrix))
  
  #Calculate $\mathrm{vec}(\psi(k_l^*))$
  psi_kstar = sp.digamma(kstar)
  print("psi_kstar = {}".format(psi_kstar))
  
  #Calculate $\mathbf{D}^{[q]}$ for $q \in [1,n]$
  ## Store the ranspose of the D matrices
  D_Matrices = []
  for i in range(num_dim):
    temp = A.copy()
    temp[:, i] = -s
    repinv = np.linalg.inv(temp)
    D_Matrices.append((repinv - Ainv).transpose())
  print("D Matrices:")
  for _ in D_Matrices: print(_)
  
  #Calculate gradients (preferably in matrix format)
  dLda = 2.0*(logM - logE)*(psi_a_dot_matrix - psi_a_matrix)
  print("dLda = {}".format(dLda))
  
  dLdb = 2.0*(logM - logE)*(psi_b_matrix - psi_b_dot_matrix)
  print("dLdb = {}".format(dLdb))
  
  dPidA = np.zeros((num_dim,num_dim))
  for l in range(num_dim): dPidA -= psi_kstar[l]*kstar[l]*D_Matrices[l]
  print("dPidA = {}".format(dPidA))
  
  #dfdA = 
  #dLdA = 2.0*(logM - logE)*( dfdA + dPidA - invA.transpose())

  #Update parameters


exit()

##Calls

#sign, logdet = np.linalg.slogdet( [D matrix stack] )


#\subsection{Prescription}
#\begin{itemize}
#{itemize}
#temize}
