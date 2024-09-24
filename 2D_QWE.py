import numpy as np

#==============================================================================#
# GRID FUNCTIONS
#==============================================================================#
def grid(n : int):
    '''Returns a square grid of n ** 2 points.'''
    out = n * []
    for i in range(n):
        out.append(n * [])
        for j in range(n):
            out[i].append((i, j))
    return out


def edgesNeumann(grid):
    '''Returns the edges of a grid according to the Neumann
    boundary conditions.'''
    n = len(grid)
    out = []
    for i in range(n):
        for j in range(n):
            if j < n - 1:
                out.append((grid[i][j], grid[i][j + 1]))
            if i < n - 1:
                out.append((grid[i][j], grid[i + 1][j]))
    return out

def edgesDirichlet(grid):
    '''Returns the edges of a grid according to the Dirichlet
    boundary conditions.'''
    n = len(grid)
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
                

def conc(grid):
    '''Returns the concatenation of the points in a grid.'''
    out = []
    for row in grid:
        for point in row:
            out.append(point)
    return out




#==============================================================================#
#                           MATRIX FUNCTIONS                                   #
#==============================================================================#
def B_Dirichlet(n : int):
    '''Returns the B matrix for a grid of n ** 2 points according to 
    the Dirichlet boundary condition.'''
    g = grid(n)
    e = edgesDirichlet(g)
    g = conc(g)
    out = np.zeros((len(e), len(g)))
    for i, (a, b) in enumerate(e):
        if a == b:
            if a[0] in [0, n - 1] and a[1] in [0, n - 1]:
                out[i][g.index(a)] = np.sqrt(2) # Corner boundary points
            else:
                out[i][g.index(a)] = 1  # Non-corner boundary points
        else:
            out[i][g.index(a)] = 1
            out[i][g.index(b)] = -1
    return out.transpose()

def B_Neumann(n : int):
    '''Returns the B matrix for a grid of n ** 2 points according to
    the Neumann boundary condition.'''
    g = grid(n)
    e = edgesNeumann(g)
    g = conc(g)
    out = np.zeros((len(e), len(g)))
    for i, (a, b) in enumerate(e):
        out[i][g.index(a)] = 1
        out[i][g.index(b)] = -1
    return out.transpose()

print(B_Dirichlet(2))

# TODO: Sort out the order of the points in the grid.

# Hamiltonian setup
from Hamiltonians.Libraries import HamiltonianEvolution as HE

n = 3
H = HE.BHamiltonian2D(n, B_Dirichlet(n), 'Dirichlet')
print(H)