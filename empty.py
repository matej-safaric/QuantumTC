# Description: This is an empty file for testing purposes
import numpy as np


#==============================================================================#
#                            Global Variables                                  #
#------------------------------------------------------------------------------#

n = 10000
a = 0.01
fps = 60

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

def f2():
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
    

#==============================================================================#
#                             Matrix exponentials                              #
#------------------------------------------------------------------------------#

from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm, logm
from numpy import pi
import matplotlib.animation as animation
from typing import Callable as func

plt.rcParams['axes.grid'] = True

def U(t):
    t = complex(t)
    return np.array([[np.cos(t * pi / 2), -np.sin(t * pi / 2)], [np.sin(t * pi / 2), np.cos(t * pi / 2)]])

def U_f(f : func, t):
    t = complex(t)
    return np.array([[f(t), -np.sqrt(1 - f(t) ** 2)], [np.sqrt(1 - f(t) ** 2), f(t)]])

def f3(f : func, range, dt):
    x = U_f(f, 0)[:,0]
    y = U_f(f, 0)[:,1]
    ts = np.arange(0, range, dt)
    fig, axs = plt.subplots(nrows=2, figsize=(8, 8))
    axs[0].set_axisbelow(True)
    axs[1].set_axisbelow(True)  
    plt.grid(visible=True, figure=fig)
    # One axis is going to plot the first complex number of each point, the other the second.
    point1a = axs[0].scatter(np.real(x[0]), np.imag(x[0]), color='blue')
    point2a = axs[0].scatter(np.real(y[0]), np.imag(y[0]), color='red')
    
    point1b = axs[1].scatter(np.real(x[1]), np.imag(x[1]), color='blue')
    point2b = axs[1].scatter(np.real(y[1]), np.imag(y[1]), color='red')
    axs[0].set(xlim=[-2, 2], ylim=[-2, 2], xlabel='Real', ylabel='Imaginary')
    axs[1].set(xlim=[-2, 2], ylim=[-2, 2], xlabel='Real', ylabel='Imaginary')

        
    def update(frame):
        x = U_f(f, ts[frame])[:,0]
        y = U_f(f, ts[frame])[:,1]
        axs[0].clear()
        axs[1].clear()
        plt.grid(visible=True, figure=fig)
        axs[0].set(xlim=[-2, 2], ylim=[-2, 2], xlabel='Real', ylabel='Imaginary')
        axs[1].set(xlim=[-2, 2], ylim=[-2, 2], xlabel='Real', ylabel='Imaginary')
        
        point1a = axs[0].scatter(np.real(x[0]), np.imag(x[0]), color='blue')
        point2a = axs[0].scatter(np.real(y[0]), np.imag(y[0]), color='red')
        
        point1b = axs[1].scatter(np.real(x[1]), np.imag(x[1]), color='blue')
        point2b = axs[1].scatter(np.real(y[1]), np.imag(y[1]), color='red')
        return point1a, point2a, point1b, point2b

    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()
    
f3(lambda t: t, pi, 0.01)
