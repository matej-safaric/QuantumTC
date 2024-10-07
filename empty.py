# Description: This is an empty file for testing purposes
import numpy as np


#==============================================================================#
#                            Global Variables                                  #
#------------------------------------------------------------------------------#

n = 10000
a = 0.01

#==============================================================================#
#                                   Main                                       #
#------------------------------------------------------------------------------#

def f1():
    for i in range(n):
        print('#==============================================================================#')
        print('h = ', n * a * np.sqrt(3) / 2)
        print('l = ', (2*n + 1) * a / 2)
        print('Difference: ', (2*n + 1) * a / 2 - n * a * np.sqrt(3) / 2)
    print('#==============================================================================#')




#==============================================================================#
#                                   3D Plots                                   #
#------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.random.rand(100, 3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

x = np.array(11 * list(range(11)))
y = np.array([11 * [i] for i in range(11)]).flatten()
from Hamiltonians.Libraries import HamiltonianEvolution as HE
z = np.array([HE.gaussian2D(i, j, 5, 5, 3) for i,j in zip(x,y)])

print(data)

ax.plot_trisurf(x,y,z)
plt.show()