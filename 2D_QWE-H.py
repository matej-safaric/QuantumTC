import numpy as np
from termcolor import colored
from Hamiltonians.Libraries import HamiltonianEvolution as HE

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 3
var = 1
t = 1000
dt = 1

#==============================================================================#
#                              GRID FUNCTIONS                                  #
#------------------------------------------------------------------------------#

def hexGrid(n : int):
    '''Returns a grid of n ** 2 points. We imagine the grid as a hexagonal
    lattice of the form
                                 0 - 1 - 2 - ...
                                / \ / \ / \ 
                               3 - 4 - 5 - ...
                                \ / \ / \ /
                                 6 - 7 - 8 - ...
                                / \ / \ / \ 
                                    ...                                 '''
    out = n * []
    for i in range(n):
        out.append(n * [])
        for j in range(n):
            out[i].append((i, j))
    return out

def edges(grid, cond = 'Neumann'):
    '''Given a hexagonal grid, returns the edges of the grid according to the
    appropriate boundary conditions.'''
    n = len(grid)
    if cond == 'Neumann':
        out = []
        for i in range(n):
            for j in range(n):
                if j < n - 1:
                    out.append((grid[i][j], grid[i][j + 1]))
                if i < n - 1:
                    out.append((grid[i][j], grid[i + 1][j]))
        return out
    elif cond == 'Dirichlet':
        out = []
        for i in range(n):
            for j in range(n):
                # If the point is on the boundary, add a self-loop.
                if i in [0, n - 1] or j in [0, n - 1]:
                    out.append((grid[i][j], grid[i][j]))
                if j < n - 1:
                    out.append((grid[i][j], grid[i][j + 1]))
                if i < n - 1:
                    out.append((grid[i][j], grid[i + 1][j]))    
        return out

