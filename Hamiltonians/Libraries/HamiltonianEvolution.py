#==============================================================================#
# This library contains functions that are used to plot the evolution of a
# quantum state under a given Hamiltonian. 

# TODO:
# - Implement an ABC on the boundary.
#==============================================================================#
# NECCESSARY LIBRARIES
from scipy.linalg import expm, logm
from qiskit.quantum_info import Operator, Statevector
import matplotlib.pyplot as plt
from numpy import arange, array
import numpy as np
import matplotlib.animation as animation
from termcolor import colored
from math import ceil

#==============================================================================#
# THE BASICS

def hamiltonian(A, t0):
    '''Given a matrix A and time t, return the Hamiltonian, according to 
    which the evolution operator is A(t) = exp(-iHt).'''
    H = 1j * logm(A) / t0
    return H

def evolveTime(H, t, psi0 : Statevector):
    '''Given a Hamiltonian H, time t, and initial state psi0, return the 
    evolved state psi(t).'''
    U = expm(-1j * H * t)
    psi_t = psi0.evolve(Operator(U))
    return psi_t




#==============================================================================#
# SINGLE QUBIT CASE

def plotQubitEvolution(H, psi0, tmax, dt):
    '''Plot the evolution of the single qubit state psi0 under 
    the Hamiltonian H.'''
    alphas = []
    betas = []
    
    ts = arange(0, tmax, dt)
    for t in ts:
        psi = evolveTime(H, t, psi0)
        try:
            alphas.append(psi.probabilities_dict()['0'])
        except KeyError:
            alphas.append(0)
        try:
            betas.append(psi.probabilities_dict()['1'])
        except KeyError:
            betas.append(0)
            
    plt.plot(ts, alphas, label='|0>', color='red')
    plt.plot(ts, betas, label='|1>', color='blue')
    plt.show()

#------------------------------------------------------------------------------#	   
# Example usage
def example1():
    A = [[0, 1], [1, 0]]
    t0 = 1
    H = hamiltonian(A, t0)
    psi0 = Statevector.from_label('0')
    plotQubitEvolution(H, psi0, 2, 0.01)
    
    
    
    
    
    
    
    
#==============================================================================#
# MULTIPLE DIMENSIONAL STATE CASE


def plotEvolution(H, psi0, tmax, dt):
    '''Plot the evolution of the state psi0 under the Hamiltonian H. Here 
    each curve describes the motion of a single vertex.'''
    probabilities = []
    n = len(psi0) // 2

    # We are not interested in the latter half of the probabilities, as they
    # correspond to the evolution of the edge states
    ts = arange(0, tmax, dt)
    for t in ts:
        psi = evolveTime(H, t, psi0)
        probabilities.append(np.sqrt(psi.probabilities()[:n]))
            
    fig, axs = plt.subplots(n, 1, sharex=True, sharey=True, clear=True)
    for i in range(n):
        axs[i].plot(ts, [p[i] for p in probabilities], label=f'|{i}>')
    plt.legend()
    plt.show()    
    
    
    
    
    
def animateEvolution(H, psi0, tmax, dt):
    '''Animates the field of each vertex under the influence of 
    a Hamiltonian [H] and given the starting state [psi0]. The 
    discretization is dictated by the time step [dt] and the
    total time [tmax].'''
    # Data preparation
    n = len(psi0) // 2 # n is the number of vertices
    m = len(psi0) # m is the size of the initial state: m = 2n + 1
    ts = arange(0, tmax, dt) # Time steps

    # Evolution
    wavefunctions = []
    wavefunctions.append(np.sqrt(psi0.probabilities()[:n]))
    for i, t in enumerate(ts):
        psi = evolveTime(H, t, psi0)
        vals = np.sqrt(psi.probabilities()[:n])
        wavefunctions.append(vals)
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    fig, ax = plt.subplots()
    x = [i / (n-1) for i in range(n)]
    wave = ax.plot(x, wavefunctions[0])[0]
    ax.set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Amplitude')
        
    def update(frame):
        y = wavefunctions[frame]
        wave.set_data(x, y)
        return wave
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1)
    plt.show()

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
    ts = arange(0, tmax, dt) # Time steps

    # Evolution
    wavefunctionsR = []
    wavefunctionsI = []
    wavefunctionsR.append(np.real(psi0.data[:n]))
    wavefunctionsI.append(np.imag(psi0.data[:n]))
    for i, t in enumerate(ts):
        psi = evolveTime(H, t, psi0)
        vals = np.sqrt(psi.probabilities()[:n])
        wavefunctionsR.append(np.real(psi.data[:n]))
        wavefunctionsI.append(np.imag(psi.data[:n]))
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = [i / (n-1) for i in range(n)]
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
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1)
    plt.show()


def animateEvolution2D(H, psi0, tmax, dt, n : int):
    '''Animates the field of each vertex under the influence of
    a Hamiltonian [H] and given the starting state [psi0]. The
    discretization is dictated by the time step [dt] and the
    total time [tmax].
    
    In this function we specify the number of vertices [n].'''
    # Data preparation
    a = int(np.sqrt(n)) # a is the number of vertices in one dimension
    m = len(psi0) # m is the size of the initial state
    ts = arange(0, tmax, dt) # Time steps

    # Evolution
    wavefunctions = []
    wavefunctions.append(np.sqrt(psi0.probabilities()[:n]))
    for i, t in enumerate(ts):
        psi = evolveTime(H, t, psi0)
        vals = np.sqrt(psi.probabilities()[:n])
        wavefunctions.append(vals)
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(wavefunctions)
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    global wave
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))    
    x = [i / (n-1) for i in range(a)]
    y = [i / (n-1) for i in range(a)]
    print(x, y)
    x, y = np.meshgrid(x, y)
    
    # Convert the wavefunctions to a 2D array
    wavefunctionsFIXED = []
    for wave in wavefunctions:
        wave = np.array(wave).reshape((a, a))
        wavefunctionsFIXED.append(wave)
        
    print(wavefunctionsFIXED[0])
    
    wave = [ax.plot_surface(x, y, wavefunctionsFIXED[0], color='b')]
    #ax.set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Amplitude')
        
    def update(frame, wave, wavefunctionsFIXED):
        z = wavefunctionsFIXED[frame]
        #print(wave)
        wave[0].remove()
        wave[0] = ax.plot_surface(x, y, z, color='b')
        return wave
    
    ax.set_zlim(0, 0.5)
    anime = animation.FuncAnimation(fig=fig, func=update, fargs=(wave, wavefunctionsFIXED),frames=(len(ts) - 1), interval=150)
    plt.show()

# Example usage
def example123():
    pass


#==============================================================================#
# INITIAL CONDITIONS

# Static initial conditions
def initCond_static1(n : int):
    mid = n // 2
    out = (mid - 1) * [0] + [1] + (n - mid) * [0] + (n + 1) * [0]
    return out

def initCond_static2(n : int):
    out = n * [0] + (n + 1) * [0]
    out[0] = 1
    return out



# Implementation of the Gaussian wave packet
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian2D(x, y, mux, muy, sigma):
    return 1 / (2 * np.pi * sigma ** 2) * np.exp(-0.5 * ((x - mux) ** 2 + (y - muy) ** 2) / sigma ** 2)

def initial_condition_gaussian(n : int, var):
    '''Returns the initial state of the system in which the first [n] elements
    represent the vertices (with a gaussian starting state) and the last n+1
    elements represent the edges (with a zero starting state). The parameter [var]
    dictates the spread of the distribution.'''
    out = []
    for i in range(n):
        out.append(gaussian(i, n//2, var))
    for i in range(n+1):
        out.append(0)
    return out

def initial_condition_gaussian2D(n : int, var, cond='Neumann'):
    '''Returns the initial state of the system in which the first [n ** 2] elements
    represent the vertices (with a gaussian starting state) and the last 2n ** 2 - 2n
    elements represent the edges (with a zero starting state). The parameter [var]
    dictates the spread of the distribution.'''
    out = []
    grid = [(i + 1, j + 1) for j in range(n) for i in range(n)]
    for (i,j) in grid:
        out.append(gaussian2D(i, j, ceil(n/2), ceil(n/2), var))
    if cond == 'Neumann':
        for i in range(2 * n ** 2 - 2 * n):
            out.append(0)
    elif cond == 'Dirichlet':
        for i in range(2 * n ** 2 + 2 * n - 4):
            out.append(0)
    return out
    
#---------------------------------------------------------------------------------#


def B(n : int):
    '''**NO BOUNDARY**: Given an integer [n], the function returns a matrix B, as defined
    in the article, for a graph with [n] vertices and [n+1] edges.'''
    out = []
    out.append([0, 1] + [0]*(n-1))
    for i in range(1, n):
        out.append([0]*i + [-1, 1] + [0]*(n-1-i))
    return array(out)

def B_Dirichlet(n : int):
    '''**DIRICHLET BOUNDARY**: Given an integer [n], the function returns a matrix B, as defined 
    in the article, for a graph with [n] vertices and [n+1] edges.'''
    out = []
    out.append([1, 1] + [0]*(n-1))
    for i in range(1, n):
        out.append([0]*i + [-1, 1] + [0]*(n-1-i))
    return array(out)

def BHamiltonian1D(n : int, B):
    '''Given an integer [n] and matrix [B], the function returns the Hamiltonian matrix 
    for a graph with [n] vertices and [n+1] edges, as dictated by the article.'''
    H = np.block([[np.zeros((n, n)), B], [np.transpose(B), np.zeros((n+1, n+1))]]) / (n+1)
    return H

def BHamiltonian2D(n : int, B, cond='Neumann'):
    '''Given an integer [n] and matrix [B], the function returns the Hamiltonian matrix 
    for a graph with [n ** 2] vertices and [2 * n ** 2 - 2 * n] edges, as dictated by the article.'''
    v = n ** 2
    if cond == 'Neumann':
        H = np.block([[np.zeros((n ** 2, n ** 2)), B], [np.transpose(B), np.zeros((2 * n ** 2 - 2 * n, 2 * n ** 2 - 2 * n))]]) / (n + 1)
    elif cond == 'Dirichlet':
        H = np.block([[np.zeros((n ** 2, n ** 2)), B], [np.transpose(B), np.zeros((2 * n ** 2 + 2 * n - 4, 2 * n ** 2 + 2 * n - 4))]]) / (n + 1)
    return H

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# Euclidean norm of a vector
def euclidean_norm(v):
    '''Given a vector v, the function returns the Euclidean norm of v.'''
    return np.sqrt(np.dot(v, v))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

# Example usage
def example2():
    init = initial_condition_gaussian(81, 3)
    init = init / euclidean_norm(init)

    H = BHamiltonian1D(81, B(81))
    animateEvolution(H, Statevector(init), 4000, 10)
    
def example2_V2():
    init = initial_condition_gaussian(81, 3)
    init = init / euclidean_norm(init)

    H = BHamiltonian1D(81, B_Dirichlet(81))
    animateEvolution_V2(H, Statevector(init), 1200000, 300)
    
# example2_V2()





#==============================================================================#
# PHASE EVOLUTION
# The following functions are used to plot the phase evolution of a qubit state


from numpy import angle
def relativePhase(v : Statevector):
    '''Return the relative phase of the qubit state v'''
    a, b = v[0], v[1]
    return angle(b) - angle(a)   

def plotPhaseEvolution(H, psi0, tmax, dt):
    '''Plot the evolution of the phase of the state psi0 under the Hamiltonian H'''
    a, b = psi0[0], psi0[1]
    thetas_a, thetas_b = [], []
    theta_rel = []
    ts = arange(0, tmax, dt)
    for t in ts:
        psi = evolveTime(H, t, psi0)
        
        thetas_a.append(angle(psi[0]))
        thetas_b.append(angle(psi[1]))
        theta_rel.append(relativePhase(psi))
            
    #plt.plot(ts, thetas_a, 'red', linestyle='dashed')
    #plt.plot(ts, thetas_b, 'blue', linestyle='dashed')
    #plt.scatter(ts, theta_rel, 'green')
    plt.plot(ts, thetas_a, 'red', linestyle='dashed', label='Phase of |0>')
    plt.plot(ts, thetas_b, 'blue', linestyle='dashed', label='Phase of |1>')
    plt.scatter(ts, theta_rel, color='green', label='Relative Phase', s=5)
    plt.legend()
    plt.show()
    
# Example usage
def example3():
    A = [[0, 1], [1, 0]]
    t0 = 1
    H = hamiltonian(A, t0)
    psi0 = Statevector.from_label('0')
    plotPhaseEvolution(H, psi0, 2, 0.001)
    
#==============================================================================#
# OTHER FUNCTIONS

