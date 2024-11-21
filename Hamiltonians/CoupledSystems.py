import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector
import ConstantHamiltonians

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to explore Hamiltonians of coupled systems
# Specifically we want to see whether it is possible to combine (in a simple
# manner) the X-gate's Hamiltonian on one subsystem, and another Hamiltonian
# on another subsystem that intersects the first one in only one quantum state.

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 10
T = 100
dt = 1

lmbda = 1



np.random.seed(10)


#==============================================================================#
#                             PREPARATION PHASE                                #
#------------------------------------------------------------------------------#

# The first order of business is to find the Hamiltonian that gives rise 
# to the X-gate

def hamiltonian(A, t0):
    '''Given a matrix A and time t, return the Hamiltonian, according to 
    which the evolution operator is A(t) = exp(-iHt).'''
    H = 1j * logm(A) / t0
    return H

X = [[0, 1], [1, 0]]
H_X = hamiltonian(X, 1)

# From the value of H_X we see that it is equal to 
#           [[-pi/2,   pi/2]
#            [ pi/2,  -pi/2]]


# Now we want to define the Hamiltonian that models the QWE
from Libraries import HamiltonianEvolution as HE

H_W = HE.BHamiltonian1D(n, HE.B_Dirichlet(n))




#==============================================================================#
#                                   CORRECTIONS                                #
#------------------------------------------------------------------------------#

# The next thing we must do is change the above Hamiltonians so they are 
# part of the total system (we add zeroes where necessary).


#--------------------------------- Correcting H_X -----------------------------#

# First we add zeroes to the existing two rows
for row in H_X:
    row = np.concatenate((row, np.zeros((n-1))))
    
# Then we add (n-1) rows with zeroes
for _ in range(n-1):
    np.append(H_X, (n+1) * [0])
    print(H_X)
    
print(len(H_X), len(H_X[0]), len(H_X[-1]))


#--------------------------------- Correcting H_W -----------------------------#

# First we add a zero to the start of each row
for row in H_W:
    row = [0] + row

# Then we add one row to the start 
H_W = [(n+1) * [0]] + H_W

print(len(H_W), len(H_W[0]), len(H_W[-1]))


















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
    
    anime = animation.FuncAnimation(fig=fig, func=update, frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()
