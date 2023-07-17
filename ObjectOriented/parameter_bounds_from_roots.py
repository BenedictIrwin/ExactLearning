from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd

# Gamma data contains a,b,c and root0 from the MMA command
# ExportString[Flatten[Table[ {a,b,c,Reduce[D[c^s Gamma[a s + b],s] == 0 && -10 < s < 10][[-1]][[2
# ]][[1]][[2]]},{a,{1/3,1/2,1,2,3/2}},{b,{-1,-1/2,-1/3,0,1/3,1/2,1}},{c,1,2}],2],"CSV"]

df = pd.read_csv('gamma_data.txt')
if(False):

    plt.plot(df.a.values,df.root0.values,'ko',label='a')
    #plt.plot(df.b.values,df.root0.values,'ko',label='b')
    #plt.plot(df.c.values,df.root0.values,'ko',label='c')
    plt.ylabel('root0')
    plt.xlabel('parameter')
    plt.legend()
    plt.show()

# For the c = 1 case.
# Do a 3D plot

df = df[df.c.values == 1]
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = df.a.values
ydata = df.b.values
zdata = df.root0.values
z_trial = (1.461632 - ydata)/xdata
ax.scatter3D(xdata, ydata, zdata)
ax.scatter3D(xdata, ydata, z_trial)
# ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('root_0')
plt.show()