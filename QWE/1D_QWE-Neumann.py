import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector
from typing import Callable as func
import matplotlib.animation as animation

#==============================================================================#
#                                   DESCRIPTION                                #
#------------------------------------------------------------------------------#

# The purpose of this code is to see what kinds of unitary transformations 
# we can get from the Hamiltonians as defined in the article by Costa,...

n = 50
k = 6
rounding = 3

fps = 30
T = 3000
dt = 3

mu = 11
var = 3

j = 25
d = 200

np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=160)
np.set_printoptions(suppress = True)


#==============================================================================#
#                                  AUX FUNCTIONS                               #
#------------------------------------------------------------------------------#

def euclidean_norm(v):
    '''Given a vector v, the function returns the Euclidean norm of v.'''
    return np.sqrt(np.dot(v, v))

def evolutionTransform(H, t):
    return np.round(np.real(expm(-1j * H * t)), rounding)

def evolutionTransform2(H, t):
    return expm(-1j * H * t)

def evolve(H, psi0 : Statevector, T, dt):
    '''Plots the time evolution of the starting state [psi0] under the
    Hamiltonian H.'''
    evolution = []
    for i, t in enumerate(np.arange(0, T, dt)):
        U = evolutionTransform2(H, t)
        U = Operator(evolutionTransform(H, t))
        psi = psi0.evolve(U)
        evolution.append(psi)
        print(colored(f'Evolution {i} out of {int(T/dt)} complete...', 'yellow'), end='\r')
    return evolution

def B_Neumann(n : int):
    out = np.zeros((n, n-1))
    out[0, 0] = 1
    out[n-1, n-2] = -1
    for i in range(1, n-1):
        out[i,i] = 1
        out[i, i-1] = -1
    return out

def BHamiltonian1D(n : int, B):
    '''Given an integer [n] and matrix [B], the function returns the Hamiltonian matrix 
    for a graph with [n] vertices and [n-1] edges, as dictated by the article.'''
    H = np.block([[np.zeros((n, n)), B], [np.transpose(B), np.zeros((n-1, n-1))]]) / (n-1)
    return H


def main1():
    H = BHamiltonian1D(n, B_Neumann(n))

    print(colored(f'#========================================================#\nThe Hamiltonian for n = {n} is:\n', 'green'))
    print(colored(H, 'green'))

    for j in range(1, k+1):
        print(colored('#------------- ', 'green'), end='')
        print(colored(f'({j} / {k})', 'red'), end='')
        print(colored(' ---------------#', 'green'))
        print(colored(f'The appropriate [expm(H, {j})] is:\n', 'green'))
        print(colored(f'{evolutionTransform(H, j)}', 'green'))
    print(colored('#========================================================#', 'green'))


#==============================================================================#
#                               Augmented Weights                              #
#------------------------------------------------------------------------------#

# Suppose we decide to change the weights of the edges from some vertex [j] to
# the right. First, we will change all the weights, then only a few, then only 
# one.

def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def initial_condition_gaussian(n : int, mu, var):
    '''Returns the initial state of the system in which the first [n] elements
    represent the vertices (with a gaussian starting state) and the last n+1
    elements represent the edges (with a zero starting state). The parameter [var]
    dictates the spread of the distribution.'''
    out = []
    for i in range(n):
        out.append(gaussian(i, mu, var))
    for i in range(n-1):
        out.append(0)
    return out


def AugB_Neumann(n : int, j : int, d : int, f : func):
    '''Given a 1D grid of [n] points, the function gives the matrix B, such
    that the edges from the [j]-th vertex onward have weights that follow
    the function f(i) up until the [j + d]-th vertex.
    
    Important: The changed edges lie between the [j]-th and [j+d]-th vertex.'''
    if j + d >= n:
        d = n - j
        
    out = np.zeros((n, n-1))
    out[0, 0] = 1
    out[n-1, n-2] = -1 if j+d != n else -f(n-1)

    for i in range(1, n-1):
        if i == j:
            out[i, i] = f(i)
            out[i, i-1] = -1
        elif i == j+d:
            out[i, i] = 1
            out[i, i-1] = -f(i-1)
        elif i not in range(j+1, j+d):
            out[i, i] = 1
            out[i, i-1] = -1
        else:
            out[i, i] = f(i)
            out[i, i-1] = -f(i-1)
    return out

print(AugB_Neumann(n, 4, 15, lambda x: 2))
print('\n')
print(B_Neumann(n))

def animateEvolution_V2(H, psi0, T, dt):
    '''Animates the field of each vertex under the influence of 
    a Hamiltonian [H] and given the starting state [psi0]. The 
    discretization is dictated by the time step [dt] and the
    total time [T].'''
    # Data preparation
    n = len(psi0) // 2 # n is the number of vertices
    ts = np.arange(0, T, dt)

    # Evolution
    wavefunctionsR = []
    #wavefunctionsI = []
    wavefunctionsR.append(np.real(psi0.data[:n]))
    #wavefunctionsI.append(np.imag(psi0.data[:n]))
    evolution = evolve(H, psi0, T, dt)
    for psi in evolution:
        wavefunctionsR.append(np.real(psi.data[:n]))
        #wavefunctionsI.append(np.imag(psi.data[:n]))
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.set_figwidth(12)
    fig.set_figheight(10)
    x = range(n)
    waveR = axs.plot(x, wavefunctionsR[0])[0]
    #waveI = axs[1].plot(x, wavefunctionsI[0])[0]
    axs.set(ylim=[-1, 1], xlabel='Position', ylabel='Real Amplitude')
    #axs[1].set(ylim=[-1, 1], xlabel='Position', ylabel='Imaginary Amplitude')
        
    def update(frame):
        yR = wavefunctionsR[frame]
        #yI = wavefunctionsI[frame]
        waveR.set_data(x, yR)
        #waveI.set_data(x, yI)
        #return [waveR, waveI]
        return [waveR]
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()


H = BHamiltonian1D(n, AugB_Neumann(n, j, d, lambda x: 2 ** (x - j)))
# H = BHamiltonian1D(n, B_Neumann(n))
init = initial_condition_gaussian(n, mu, var)
psi0 = Statevector(init / euclidean_norm(init))

animateEvolution_V2(H, psi0, T, dt)