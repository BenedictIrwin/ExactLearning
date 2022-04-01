import numpy as np
import sys
import random
import math
import matplotlib.pyplot as plt

#file_name = sys.argv[1]

## Get the means and standard deviations of the array

## Read in a random matrix of observations of a fixed length

def mp_dist(gamma,x):
  return np.sqrt(((1+np.sqrt(gamma))**2-x)*(x-(1-np.sqrt(gamma))**2))/(2.0*math.pi*gamma*x)


fixed_length = 200
num_obs = 5000

matrix = np.random.normal(loc=0.0,scale=1.0,size = [num_obs,fixed_length])
matrix_transpose = matrix.T

mu = [ np.mean(i) for i in matrix_transpose]
sigma = [ np.std(i) for i in matrix_transpose]




matrix_norm = np.array([ np.divide(matrix[i]-mu,sigma) for i in range(len(matrix)) ])

correlation_matrix = np.matmul(matrix_norm.T,matrix_norm)/num_obs


print(correlation_matrix.shape)

w, v = np.linalg.eig(correlation_matrix)

#print(w)
#print(v) 

#for i in w: print(i)

w = np.real(w)

plt.hist(w,density=True)  # arguments are passed to np.histogram
plt.title("Sampled Histogram with Theoretical MP Distribution")

gamma = fixed_length/num_obs

x = np.linspace((1.0-np.sqrt(gamma))**2,(1.0+np.sqrt(gamma))**2,100)
y = mp_dist(gamma,x)
plt.plot(x,y,'-',color='red')

plt.show()
