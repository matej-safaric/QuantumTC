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

n = 60
k = 6
rounding = 3

fps = 30
T = 3000
dt = 12

# Gaussian
mu1 = 11
var1 = 3

# Richer
mu2 = 11
var2 = 16

j = 30
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
    '''Returns the time evolution of the starting state [psi0] under the
    Hamiltonian H.'''
    evolution = []
    for i, t in enumerate(np.arange(0, T, dt)):
        U = evolutionTransform2(H, t)
        U = Operator(evolutionTransform2(H, t))
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

def derivative_of_gaussian(x, mean=0.0, std_dev=1.0):
    """
    Computes the derivative of the Gaussian function at a given point.

    Parameters:
        x (float or array-like): The point(s) where the derivative is computed.
        mean (float): The mean of the Gaussian distribution.
        std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
        float or array-like: The derivative of the Gaussian function.
    """
    # Normalized variable
    z = (x - mean) / std_dev
    # Derivative of Gaussian formula
    derivative = -z * np.exp(-0.5 * z**2) / (std_dev * np.sqrt(2 * np.pi))
    return derivative

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

def initial_condition_gaussian_moving(n : int, mu, var, direc=1):
    '''Returns the initial state of a quantum system, such that the 
    generated wave propagation is a gaussian packet moving in the 
    direction specified by [dir], whose value is either 1 or -1.'''
    outV = []
    for i in range(n):
        outV.append(gaussian(i, mu, var))
    outV = [i / euclidean_norm(outV) for i in outV]
    outE = []
    for i in range(n-1):
        outE.append(direc * 1j * gaussian(i + 0.5, mu, var))
    print(colored(outE, 'cyan'))
    outE = [i / euclidean_norm(outE) for i in outE]
    return outV + outE





def richer(t, f0=1.0, mean=0.0, std_dev=1.0):
    """
    Computes the Richer wavelet function with optional mean and standard deviation.

    Parameters:
        t (array-like): Time values.
        f0 (float): Central frequency of the wavelet.
        mean (float): Mean of the wavelet distribution (time shift).
        std_dev (float): Standard deviation (scaling factor).

    Returns:
        array-like: Richer wavelet values.
    """
    # Apply mean and standard deviation to the time values
    normalized_t = (t - mean) / std_dev
    
    # Adjusted factor to create the wavelet
    term = (np.pi * f0 * normalized_t)
    wavelet = (1 - 2 * term**2) * np.exp(-term**2)
    return wavelet


def initial_condition_richer(n : int, mu, var):
    out = []
    for i in range(n):
        out.append(richer(i, 1.0, mu, var))
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
        
    out = np.zeros((n, n-1), dtype=complex)
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

# print(AugB_Neumann(n, 4, 15, lambda x: 2))
# print('\n')
# print(B_Neumann(n))

def animateEvolution_V2(H, psi0, T, dt, glb_phase_shift=0, imag : bool=False):
    '''Animates the field of each vertex under the influence of 
    a Hamiltonian [H] and given the starting state [psi0]. The 
    discretization is dictated by the time step [dt] and the
    total time [T].
    
    The glb_phase_shift is an optional parameter that shifts the 
    global phase for each datapoint by a fixed constant.'''
    # Data preparation
    n = len(psi0) // 2  + 1# n is the number of vertices
    ts = np.arange(0, T, dt)
    # Evolution
    wavefunctionsVr = []
    wavefunctionsVi = []
    
    wavefunctionsEr = []
    wavefunctionsEi = []
    
    # Vertex data
    wavefunctionsVr.append(np.real(psi0.data[:n]))
    wavefunctionsVi.append(np.imag(psi0.data[:n]))
    # Edge data
    wavefunctionsEr.append(np.real(psi0.data[n:]))
    wavefunctionsEi.append(np.imag(psi0.data[n:]))
    
    evolution = evolve(H, psi0, T, dt)
    for i, psi in enumerate(evolution):
        if not psi.is_valid():
            print(colored(f'Non-valid state at time step {i} / {len(evolution)}', 'red'))
        wavefunctionsVr.append(np.real(psi.data[:n] * np.exp(1j * glb_phase_shift)))
        wavefunctionsVi.append(np.imag(psi.data[:n] * np.exp(1j * glb_phase_shift)))
        wavefunctionsEr.append(np.real(psi.data[n:] * np.exp(1j * glb_phase_shift)))
        wavefunctionsEi.append(np.imag(psi.data[n:] * np.exp(1j * glb_phase_shift)))
    print(colored('Evolutions completed. Plotting...\n', 'green'))
           
    # Plotting
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_figwidth(12)
    fig.set_figheight(10)
    xV = range(n)
    xE = [i + 0.5 for i in range(n-1)]
    waveVr = axs[0].plot(xV, wavefunctionsVr[0])[0]
    line = axs[0].plot(xV, np.zeros(len(xV)), linestyle='dashed')[0]
    waveEr = axs[1].plot(xE, wavefunctionsEr[0])[0]
    
    if imag:
        waveVi = axs[0].plot(xV, wavefunctionsVi[0])[0]
        waveEi = axs[1].plot(xE, wavefunctionsEi[0])[0]
    axs[0].set(ylim=[-1, 1], xlabel='Position', ylabel='Vertex Amplitude')
    axs[1].set(ylim=[-1, 1], xlabel='Position', ylabel='Edge Amplitude')
        
    def update(frame):
        yVr = wavefunctionsVr[frame]
        yEr = wavefunctionsEr[frame]
        waveVr.set_data(xV, yVr)
        waveEr.set_data(xE, yEr)
        
        if imag:
            yVi = wavefunctionsVi[frame]
            yEi = wavefunctionsEi[frame]
            waveVi.set_data(xV, yVi)
            waveEi.set_data(xE, yEi)
        #return [waveR, waveI]
        return [waveVr, waveVi, waveEr, waveEi] if imag else [waveVr, waveEr]
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()


def snapshot(H, psi0, t):
    '''The function plots the state of the system given by H and psi0 at time t.'''
    U = Operator(evolutionTransform2(H, t))
    psi = psi0.evolve(U)
    
    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.set_figwidth(12)
    fig.set_figheight(10)
    x = range(n)
    axs.plot(x, np.real(psi.data[:n]))
    axs.plot(x, np.zeros(len(x)), linestyle='dashed')
    axs.set(ylim=[-1, 1], xlabel='Position', ylabel='Real Amplitude')
    plt.show()










#==============================================================================#
#                               DIFFERENT CASES                                #
#------------------------------------------------------------------------------#

init1 = initial_condition_gaussian(n, mu1, var1)
init2 = initial_condition_richer(n, mu2, var2)
init3 = initial_condition_gaussian_moving(n, mu1, var1)

psi0_1 = Statevector(init1 / euclidean_norm(init1))
psi0_2 = Statevector(init3 / euclidean_norm(init3))


# 1. Exponential dampening

B1 = AugB_Neumann(n, j, d, lambda x: 2 ** (x - j - 10))
H1 = BHamiltonian1D(n, B1)
# H = BHamiltonian1D(n, B_Neumann(n))



# 2. Constant dampening

B2 = AugB_Neumann(n, j, d, lambda _: 2)
H2 = BHamiltonian1D(n, B2)



# 3. Different weight at the border

B3 = AugB_Neumann(n, j, d, lambda _: 1j)
H3 = BHamiltonian1D(n, B3)



# 4. Imaginary values in B
def randComplex(shape : int):
    '''Generates [shape] random complex numbers from the unit disk in the 
    form of a numpy array.'''
    return np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))

q = randComplex(1)[0]
B4 = AugB_Neumann(n, j, d, lambda x: q)
H4 = BHamiltonian1D(n, B4)

# print(colored('B4:\n', 'green'), AugB_Neumann(6, 3, 11, lambda x: q), '\n\n')
# print(colored('B4 * B4.H:\n', 'green'), np.matmul(B4, B4.conj().T), '\n\n')
# print(colored('B4.H * B4:\n', 'green'), np.matmul(B4.conj().T, B4))




# 5. Linear dampening

B5 = AugB_Neumann(n, j, d, lambda x: (x-j+2))
H5 = BHamiltonian1D(n, B5)

#snapshot(H1, psi0, 2 * T / 3)
animateEvolution_V2(H5, psi0_2, T, dt, glb_phase_shift=-np.pi / 4, imag=True)