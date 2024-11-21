import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to experiment with a rudimentary version
# of the TFSF boundary in 2 dimensions. The theoretical background is 
# presented in the Obsidian note Report 7-11-2024



#==============================================================================#
#                              GLOBAL VARIABLES                                #
#------------------------------------------------------------------------------#

# Number of points in the discretization (the grid is then nxn)
n = 20 

# Distance between points
a = 1 

# Time discretization
T = 5000
dt = 5

# FPS of the animation (where applicable)
fps = 30 

# Spread and expected value of the Gaussian/Richer wavelet
mu = n // 2
var = 3

# Side length of scatterer (square shaped)
m = n // 3

# Lower left corner of scatterer position
x_S = n // 3
y_S = n // 3



#==============================================================================#
#                             IMPORTED FUNCTIONS                               #
#------------------------------------------------------------------------------#

def evolutionFunction(H, t):
    '''Returns the unitary matrix for the Hamiltonian H at time t.'''
    return expm(-1j * H * t)

def euclidean_norm(v):
    '''Given a vector in C, the function calculates its modulus.'''
    return np.sqrt(np.sum(np.abs(v)**2))


# The following is an imported function from HamiltonianEvolution, which we 
# will be taking inspiration from.

def animateEvolution_V2(H, psi0, tmax, dt):
    '''Animates the field of each vertex under the influence of 
    a Hamiltonian [H] and given the starting state [psi0]. The 
    discretization is dictated by the time step [dt] and the
    total time [tmax].
    
    This function is different from the previous one in the 
    sense, that it goes further with modelling whole complex
    values of each vertex, instead of just their magnitudes.'''
    # Data preparation
    n = len(psi0) // 2 # n is the number of vertices
    m = len(psi0) # m is the size of the initial state: m = 2n + 1
    ts = np.arange(0, tmax, dt) # Time steps

    # Evolution
    wavefunctionsR = []
    wavefunctionsI = []
    wavefunctionsR.append(np.real(psi0.data[:n]))
    wavefunctionsI.append(np.imag(psi0.data[:n]))
    for i, t in enumerate(ts):
        psi = psi0.evolve(Operator(evolutionFunction(H, t)))
        vals = np.sqrt(psi.probabilities()[:n])
        wavefunctionsR.append(np.real(psi.data[:n]))
        wavefunctionsI.append(np.imag(psi.data[:n]))
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = [i * a / (n-1) for i in range(n)]
    waveR = axs[0].plot(x, wavefunctionsR[0])[0]
    waveI = axs[1].plot(x, wavefunctionsI[0])[0]
    axs[0].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Real Amplitude')
    axs[1].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Imaginary Amplitude')
        
    def update(frame):
        yR = wavefunctionsR[frame]
        yI = wavefunctionsI[frame]
        waveR.set_data(x, yR)
        waveI.set_data(x, yI)
        return [waveR, waveI]
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()

# The following are some functions that correspond to the implementation of FDTD like
# for example the Richer wavelet initial condition and the required Hamiltonian.

def richer(x, mu, sigma):
    return 2 / (np.sqrt(3 * sigma) * (np.pi ** 0.25)) * (1 - ((x - mu) / sigma) ** 2) * np.exp((- (x - mu) ** 2) / (2 * sigma ** 2))
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * np.sin(2 * np.pi * x)

def initRicher(n : int, mu, var):
    '''Returns the initial state of the system in which the first [n] elements
    represent the vertices (with a gaussian starting state) and the last n+1
    elements represent the edges (with a zero starting state). The parameter [var]
    dictates the spread of the distribution.'''
    out = []
    for i in range(n):
        out.append(richer(i, mu, var))
    for i in range(n+1):
        out.append(0)
    return Statevector(out / euclidean_norm(out))




def B_Dirichlet(n : int):
    '''**DIRICHLET BOUNDARY**: Given an integer [n], the function returns a matrix B, as defined 
    in the article, for a graph with [n] vertices and [n+1] edges.'''
    out = []
    out.append([1, 1] + [0]*(n-1))
    for i in range(1, n):
        out.append([0]*i + [-1, 1] + [0]*(n-1-i))
    return np.array(out)

def BHamiltonian1D(n : int, B):
    '''Given an integer [n] and matrix [B], the function returns the Hamiltonian matrix 
    for a graph with [n] vertices and [n+1] edges, as dictated by the article.'''
    H = np.block([[np.zeros((n, n)), B], [np.transpose(B), np.zeros((n+1, n+1))]]) / (n+1)
    return H

