import numpy as np
import sys
import random


file_name = sys.argv[1]

matrix = []
tags = []
with open(file_name) as f:
  for line in f:
    line=line.strip().split(":")
    vec = line[1].split(",")
    vec.pop(-1)
    vec = [ float(i) for i in vec]
    matrix.append(vec)
    tags.append(line[0])

tages = np.array(tags)
matrix = np.array(matrix)

## Get the means and standard deviations of the array

matrix_transpose = matrix.T

mu = [ np.mean(i) for i in matrix_transpose]
sigma = [ np.std(i) for i in matrix_transpose]




matrix_norm = np.array([ np.divide(matrix[i]-mu,sigma) for i in range(len(matrix)) ])

correlation_matrix = np.matmul(matrix_norm,matrix_norm.T)/99998.0

#print(correlation_matrix)

w, v = np.linalg.eig(correlation_matrix)

#print(w)
#print(v) 

for i in w: print(i)
