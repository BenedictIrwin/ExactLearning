import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt


with open("test_1d_distributions.txt","r") as f:
  header = f.readline()
  for line in f:
    name, min, max, python, mma, moments = line.split(",")
    min = float(min)
    max = float(max)
    print(name, min, max, python, mma, moments)
    x = np.linspace(min,max,100)
    f = eval("lambda x :"+ python)
    plt.plot(x,f(x))
    plt.show()
