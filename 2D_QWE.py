import numpy as np
from termcolor import colored
from Hamiltonians.Libraries import HamiltonianEvolution as HE


#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 9
var = 2
t = 100
dt = 1

#==============================================================================#
#                              GRID FUNCTIONS                                  #
#------------------------------------------------------------------------------#

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

#==============================================================================#
#                           SIMULATION FUNCTIONS                               #
#------------------------------------------------------------------------------#


def simulation2D(cond):
    '''Given a specified boundary condition, the function simulates the 
    2D wave equation.'''
    
    # Setting up Hamiltonian
    print(colored('Setting up Hamiltonian...', 'blue'), end='\r')
    if cond == 'Neumann':
        H = HE.BHamiltonian2D(n, B_Neumann(n), 'Neumann')
    else:
        H = HE.BHamiltonian2D(n, B_Dirichlet(n), 'Dirichlet')

    print(colored('Hamiltonian set up...    \n', 'green'))
    print(colored('The Hamiltonian is:', 'light_blue'))
    print(colored(f'{H}\n', 'light_blue'))


    # Initial state
    print(colored('Setting up initial condition...', 'blue'), end='\r')

    init = HE.initial_condition_gaussian2D(n, var, 'Dirichlet' if cond == 'Dirichlet' else 'Neumann')
    psi0 = HE.Statevector(init / HE.euclidean_norm(init))

    print(colored('Initial condition set up...', 'green'))
    print(colored('The initial condition is:', 'light_blue'))
    print(colored(f'{psi0}\n', 'light_blue'))
    

    # Evolve & animate
    HE.animateEvolution2D_V2(H, psi0, t, dt, n**2)
    
    
simulation2D('Neumann')