import matplotlib.pyplot as plt
import numpy as np

import os

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]

for i in onlyfiles:
  if(".npy" in i and "pot_" in i):
    ##This is a potential
    pot_name = i.strip()
    wfn_name = i.replace("pot_","wfn_")
    print(pot_name)
    print(wfn_name)
    tag = i.replace("pot_","").replace(".npy","")

    pot = np.load(pot_name)
    wfn = np.load(wfn_name)
    plt.plot(np.minimum(pot,0),wfn, label = tag)

plt.legend()
plt.show()

for i in onlyfiles:
  if(".npy" in i and "pot_" in i):
    ##This is a potential
    pot_name = i.strip()
    wfn_name = i.replace("pot_","wfn_")
    spc_name = i.replace("pot_","spc_")
    print(pot_name)
    print(wfn_name)
    tag = i.replace("pot_","").replace(".npy","")

    pot = np.load(pot_name)
    wfn = np.load(wfn_name)
    spc = np.load(spc_name)
    #plt.plot(np.minimum(pot,0),wfn, label = tag)
    plt.plot(spc,wfn, label = tag)
    plt.plot(spc,np.minimum(pot,0), label = tag)

plt.legend()
plt.show()
