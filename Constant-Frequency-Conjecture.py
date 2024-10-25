from Hamiltonians import ConstantHamiltonians as ch
from scipy import fft
import numpy as np

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to explore, whether the constant frequency 
# conjecture holds. It states that for a quantum system, whose Hamiltonian is
# constant, the sinusoidal behavior of each possible state will have the same
# global frequency.

# The code generates a random Hamiltonian, evolves a random initial state under
# it, and then plots the evolution of the system. It then takes the Fourier
# transform of the evolution of each state, and plots the frequency spectrum
# of each state. 

#==============================================================================#
#                               GLOBAL VARIABLES                               #
#------------------------------------------------------------------------------#

n = 3
T = 10
dt = 0.1
# Number of simulations to be done:
simNum = 10 

#==============================================================================#
#                                    MAIN                                      #
#------------------------------------------------------------------------------#

#----------------------------- Generating the Data ----------------------------#

simulations = []
for i in range(simNum):
    # Generate a random Hamiltonian:
    H = ch.randH(n)
    # Generate a random initial state:
    psi0 = ch.randState(n)
    # Evolve the system:
    evolution = ch.evolve(H, psi0, T, dt)
    # Reshape the data so that each row corresponds to a state:
    evolution = np.array(evolution).T
    simulations.append(evolution)
    
#----------------------------- Fourier Transform ------------------------------#
for sim in simulations:
    for state in sim:
        state = fft.fft(state)
        print(state)
    