import numpy as np
from termcolor import colored
from Hamiltonians.Libraries import HamiltonianEvolution as HE
import matplotlib.pyplot as plt
import matplotlib as mpl

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to simulate the evolution of a quantum system, 
# behaving as a wave in 1D with initial conditions set to zero and an injection
# of energy at the edge of the grid.

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 11
var = 1.5
t = 1000
dt = 10
a = 0.5

#==============================================================================#
#                              Initial conditions                              #
#------------------------------------------------------------------------------#

# Initial state.
psi0 = np.array([1] + [0] * n)


#--------------------------------- Hamiltonian --------------------------------#

# H_in = 