import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from Libraries import HamiltonianEvolution as HE
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to see, which kinds of functions we can model
# while only using constant-valued Hamiltonians. Our approach is simple:
# We will try a bunch of different constant valued Hamiltonians and observe
# the resulting evolution of the system.

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#


n = 3
T = 10
dt = 0.1
# Number of simulations to be done:
simNum = 10 


#==============================================================================#
#                            AUXILIARY FUNCTIONS                               #
#------------------------------------------------------------------------------#

def isSkewHermitian(H):
    '''Checks whether a matrix is skew-hermitian.'''
    return np.allclose(H, -H.conj().T, atol=1e-8)

def isUnitary(U):
    '''Checks whether a matrix is unitary.'''
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-8)

def euclidean_norm(v):
    '''Given a vector in C, the function calculates its modulus.'''
    return np.sqrt(np.sum(np.abs(v)**2))

#==============================================================================#
#                                SETTING UP                                    # 
#------------------------------------------------------------------------------#

# If we want the result of a matrix exponential to be unitary, the Hamiltonian
# needs to be skew-hermitian. Hence we will be generating random skew-hermitian
# matrices and then exponentiating them to see what happens.

# IMPORTANT: Skew-hermitian matrices have purely imaginary values on the 
# diagonal. Thus, our Hamiltonians will have purely real numbers on the 
# diagonal. This is due to the multiplication by i in the exponentiation.

#----------------------------- Generating Hamiltonians ------------------------#

def randComplex(shape : int):
    '''Generates [shape] random complex numbers from the unit disk in the 
    form of a numpy array.'''
    return np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))

# The following function generates random Hamiltonians for us.
def randH(n : int):
    '''Generates a random matrix H, such that iHt will be a skew-hermitian 
    matrix of size n x n.'''
    A = np.zeros((n, n), dtype = complex)
    for i in range(n):
        for j in range(i + 1, n):
            if i==j:
                A[i,j] = np.random.uniform(-1, 1)
            else:
                A[i,j] = randComplex(1)[0]
                A[j,i] = np.conjugate(A[i,j])
    return A





#---------------------------- Exponentiating Hamiltonians ---------------------#

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

def plotEvolution(evolution, n : int, i=0, optionalName : str = ''):
    '''Plots the evolution of the system.'''
    evolution = np.array(evolution)
    fig, axs = plt.subplots(nrows=1)
    for j in range(n):
        plt.plot(np.arange(0, T, dt), np.abs(evolution[:,j])**2, label = f'|{j}>')
    plt.legend()
    plt.savefig(f"Hamiltonians/Plots/evolution{i}{'-' + optionalName if optionalName != '' else ''}.png")
    axs.clear()
    
    
    
#==============================================================================#
#                                  SIMULATION                                  #
#------------------------------------------------------------------------------#

# The following is the meat of this code.

#--------------------------- Fixed initial condition --------------------------#

# We will now generate a bunch of random Hamiltonians and see what happens
# when the initial condition is set to |0>.
for i in range(simNum):
    H = randH(n)
    evolution = evolve(H, Statevector([1] + (n-1) * [0]), T, dt)
    print(colored(f'Simulation {i+1} of {simNum} complete.                      ', 'green'), end='\r')
    plotEvolution(evolution, n, i, 'fixedInitialCondition')
    

print(colored('\n\n#=====================================================================#\n\n', 'green'))


#--------------------------- Random initial condition -------------------------#

# Now we will fix a random Hamiltonian and apply it to a bunch of random
# initial conditions.

H = randH(n)
for i in range(simNum):
    rand = randComplex(n)
    psi0 = Statevector(rand / euclidean_norm(rand))
    evolution = evolve(H, psi0, T, dt)
    print(colored(f'Simulation {i+1} of {simNum} complete.                      ', 'green'), end='\r')
    plotEvolution(evolution, n, i, 'randomInitialCondition')

