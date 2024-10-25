from Hamiltonians import ConstantHamiltonians as ch
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

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

n = 5
T = 10
dt = 0.01
# Number of simulations to be done:
simNum = 3

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
    N = len(sim[0])
    fig, axs = plt.subplots(nrows=2)
    for state in sim:
        # Take the probabilities of each state
        stateP = (np.abs(state) ** 2)
        axs[0].plot(np.arange(0, T, dt), stateP)
        # Fourier transform of each state's behavior:
        freq = fft.fftfreq(N, dt)[:N//2]
        # Limit the frequency to under 1 (there is nothing after that):
        freqUnder1 = freq[freq < 1]
        stateFFT = (fft.fft(stateP)[:N//2])[:len(freqUnder1)]
        axs[1].plot(freqUnder1, np.abs(stateFFT))
    plt.grid()
    plt.show()