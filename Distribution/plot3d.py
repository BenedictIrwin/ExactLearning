# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.rcParams['axes.facecolor'] = (255/255,242/255,204/255)


fig = plt.figure()
fig.patch.set_facecolor((255/255,242/255,204/255))
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0.3, 4, 0.1)
Y = np.arange(0.3, 4, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-X*Y/(X+Y))/(X+Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0,1.33)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
