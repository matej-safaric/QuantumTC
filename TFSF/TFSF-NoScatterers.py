import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# WARNING: THIS IS NOT USEFUL IN ANY WAY!

# The purpose of this code is to implement the TFSF boundary in the special 
# case where there are no scatterers present in our domain. We are merely 
# adding another layer of vertices that represents the SF part of the 
# simulation.
# The theoretical background is given in the Obsidian file: Report 4-11-2024

#==============================================================================#
#                              GLOBAL VARIABLES                                #
#------------------------------------------------------------------------------#

# Number of points in the discretization
n = 40 

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

# The following code is simply the culmination of all of the above into a simple 1D simulation

def example():
    init = initRicher(n, mu, var)
    #plt.plot(range(n), init[:n])

    H = BHamiltonian1D(n, B_Dirichlet(n))
    animateEvolution_V2(H, Statevector(init), 500, 5)
    
# example()





#==============================================================================#
#                                   MAIN                                       #
#------------------------------------------------------------------------------#

# Here we will add another layer to our discretization, that being the SF
# layer. Let's denote the point at which the TFSF boundary is placed with [m].

m = n // 2

ts = np.arange(0, T, dt)

#-------------------------------- SF Layer ------------------------------------#
# We initialise the SF layer as an array of zeroes. This is because we 
# expect there to be no scattered field yet at the start of the simulation. 

SF = [[0] * m for _ in ts]
print(colored('SF layer prepared...', 'green'))

#-------------------------------- TF Layer ------------------------------------#
# The TF layer will simply be a simulation of the wave equation as we know it.
# Despite the TF layer actually only being the vertices from m to n, we will 
# be programming the wave equation on all n vertices.

#- - - - - - - - - - - - - - - - Setting up - - - - - - - - - - - - - - - - - -#
psi0 = initRicher(n, mu, var)
H = BHamiltonian1D(n, B_Dirichlet(n))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

TF = []
for i, t in enumerate(ts):
    psi = psi0.evolve(Operator(evolutionFunction(H, t)))
    TF.append(list(np.real(psi.data[:n])))
    print(colored(f'Evolution {i} of {len(ts)} completed.', 'cyan'), end='\r')
print(colored('Evolutions completed...', 'green'))


#---------------------------------- TFSF --------------------------------------#
# Here we put all the layers together in one plot/animation.

def TFSF():    
    fig, axs = plt.subplots(3, 1, sharex=True)
    x = [i * a / (n-1) for i in range(n)]
    SF_plot = axs[0].plot(x[:m], SF[0])[0]
    TF_plot = axs[1].plot(x, TF[0])[0]
    print(SF[0])
    print(TF[0])
    
    TFSF_plot = axs[2].plot(x, SF[0][:m] + TF[0][m:])[0]
    axs[0].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='SF-layer')
    axs[1].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='TF-layer')
    axs[2].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='TFSF')
        
    def update(frame):
        y1 = SF[frame]
        y2 = TF[frame]
        y3 = SF[frame][:m] + TF[frame][m:]
        SF_plot.set_data(x[:m], y1)
        TF_plot.set_data(x, y2)
        TFSF_plot.set_data(x, y3)
        return [SF_plot, TF_plot, TFSF_plot]
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()
    
TFSF()