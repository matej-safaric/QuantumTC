import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from termcolor import colored
from qiskit.quantum_info import Operator, Statevector

#==============================================================================#
#                                   DESCRIPTION                                #
#------------------------------------------------------------------------------#

# The purpose of this code is to see what kinds of unitary transformations 
# we can get from the Hamiltonians as defined in the article by Costa,...

n = 6
k = 6
rounding = 3

np.get_printoptions()['linewidth']
np.set_printoptions(linewidth=160)
np.set_printoptions(suppress = True)


#==============================================================================#
#                                  AUX FUNCTIONS                               #
#------------------------------------------------------------------------------#

def evolutionTransform(H, t):
    return np.round(np.real(expm(-1j * H * t)), rounding)

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
