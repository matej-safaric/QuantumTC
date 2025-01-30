# Description: This is an empty file for testing purposes
import numpy as np
from termcolor import colored


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
    
# f3(lambda t: t, pi, 0.01)






#==============================================================================#
#                             Adjacency matrices                               #
#------------------------------------------------------------------------------#


# Let's set up some code that makes a discrete mesh graph with edges going
# in both directions for every pair of vertices while also having different 
# weights

# We are starting off with a 1-dimensional mesh of length n.

n = 5

l = 1
r = 2

def B_bidirec(n, l, r):
    out1 = np.zeros((n, n-1))
    for i in range(n-1):
        out1[i, i] = r
        out1[i+1, i] = -r
    out2 = np.zeros((n, n-1))
    for i in range(n-1):
        out2[i,i] = -l
        out2[i+1, i] = l
    return np.block([out1, out2])

def Ham(B):
    (r, c) = np.shape(B)
    return np.block([[np.zeros((r,r)), B], [B.T.conj(), np.zeros((c,c))]])

# print(np.matmul(Ham(B_bidirec(n,l,r)), Ham(B_bidirec(n,l,r))))




#==============================================================================#
#                     Testing a weird imaginary Hamiltonian                    #
#------------------------------------------------------------------------------#

H = np.array([[0, np.sqrt(2) * 1j], [np.sqrt(2) * -1j, 0]])
H1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])





# The following is an excerpt of the code from the file ConstantHamiltonians.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def evolutionFunction(H, t):
    '''Returns the unitary matrix for the Hamiltonian H at time t.'''
    return expm(-1j * H * t)

def evolve(H, psi0 : Statevector, T, dt):
    '''Plots the time evolution of the starting state [psi0] under the
    Hamiltonian H.'''
    evolution = []
    for i, t in enumerate(np.arange(0, T, dt)):
        U = evolutionFunction(H, t)
        U = Operator(evolutionFunction(H, t))
        psi = psi0.evolve(U)
        evolution.append(psi)
        print(colored(f'Evolution {i} out of {int(T/dt)} complete...', 'yellow'), end='\r')
    return evolution


#----------------------------- Plotting the evolution -------------------------#
# The following function plots the evolution of the system.
# It actually only plots the probability of each state, which 
# can be safely ignored, since the purpose of this code is to
# see whether all quantum systems follow wave-like behaviour.


T = 20
dt = 0.01

def plotEvolution(evolution, n : int, i=0, optionalName : str = ''):
    '''Plots the evolution of the system.'''
    evolution = np.array(evolution)
    fig, axs = plt.subplots(nrows=1)
    for j in range(n):
        plt.plot(np.arange(0, T, dt), np.abs(evolution[:,j])**2, label = f'|{j}>')
    plt.legend()
    plt.savefig(f"Hamiltonians/Plots/ConstantHamiltonians/evolution{i}{'-' + optionalName if optionalName != '' else ''}.png")
    axs.clear()
    
    
def euclidean_norm(v):
    '''Given a vector in C, the function calculates its modulus.'''
    return np.sqrt(np.sum(np.abs(v)**2))

def randComplex(shape : int):
    '''Generates [shape] random complex numbers from the unit disk in the 
    form of a numpy array.'''
    return np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))

def randState(n):
    '''Generates a random statevector of size n.'''
    rand = randComplex(n)
    return Statevector(rand / euclidean_norm(rand))

# plotEvolution(evolve(H, randState(2), T, dt), 2, optionalName='WUBBAWUBBA')
plotEvolution(evolve(H1, randState(3), T, dt), 3, optionalName='WUBBAWUBBA1')