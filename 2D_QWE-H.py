import numpy as np
from termcolor import colored
from Hamiltonians.Libraries import HamiltonianEvolution as HE
from matplotlib import pyplot as plt
from matplotlib import animation
import time
from qiskit import Statevector

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 3
a = 1
var = 1
t = 1000
dt = 1

#==============================================================================#
#                              OTHER FUNCTIONS                                 #
#------------------------------------------------------------------------------#

def isGreater(vertex1, vertex2):
    '''Returns True if vertex1 is greater than vertex2, and False otherwise.'''
    if vertex1[0] > vertex2[0]:
        return True
    elif vertex1[0] == vertex2[0] and vertex1[1] > vertex2[1]:
        return True
    return False

#==============================================================================#
#                              GRID FUNCTIONS                                  #
#------------------------------------------------------------------------------#

def hexGrid(n : int):
    '''Returns a grid of n ** 2 points. We imagine the grid as a hexagonal
    lattice of the form in the example below.'''
    #                             0 - 1 - 2 - ...
    #                            / \ / \ / \ 
    #                           3 - 4 - 5 - ...
    #                            \ / \ / \ /
    #                             6 - 7 - 8 - ...
    #                            / \ / \ / \ 
    #                                ...                                 
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




#==============================================================================#
#                               MATRIX FUNCTIONS                               #
#------------------------------------------------------------------------------#

def D(n : int, cond='Neumann'):
    '''Returns the out-degree matrix of the grid.'''
    out = np.zeros((n ** 2, n ** 2), dtype=int)
    if cond == 'Neumann':
        for i in range(n ** 2):
            if (i + 1) % (2 * n) == 0 or (i + 1) % (2 * n) == 1:
                out[i][i] = 5 
            else:
                out[i][i] = 6
    elif cond == 'Dirichlet':
        raise NotImplementedError
    return out



def edgeTemplate(vertex, n : int, cond = 'Neumann'):
    '''Returns the edges going out of the vertex according to one of 
    the possible templates, depending on the position of the vertex.
    
    An edge is of the form ((vertex1, vertex2), weight).
    Edges going out of a given vertex are added in a counter-clockwise 
    fashion starting with the edge going to the left of the vertex.
    
    The size of the grid is n x n, where n is an ODD INTEGER.'''
    # TODO: PLEASE FOR GOD'S SAKE, MAKE THIS FUNCTION MORE EFFICIENT.
    
    if cond == 'Neumann':
        if n % 2 != 1:
            raise ValueError('The size of the grid must be an odd integer.')
        i, j = vertex
        # Top vertices and bottom vertices:
        if i == 0:
            if j in range(1, n - 1):
                return [((vertex, (i, j - 1)), 1),
                        ((vertex, (i + 1, j)), 2),
                        ((vertex, (i + 1, j + 1)), 2),
                        ((vertex, (i, j + 1)), 1)]
            elif j == 0:
                return [((vertex, (i + 1, j)), 2),
                        ((vertex, (i + 1, j + 1)), 2),
                        ((vertex, (i, j + 1)), 1)]
            elif j == n - 1:
                return [((vertex, (i, j - 1)), 2),
                        ((vertex, (i + 1, j)), 4)]
        elif i == n - 1:
            if j in range(1, n - 1):
                return [((vertex, (i, j - 1)), 1),
                        ((vertex, (i, j + 1)), 1),
                        ((vertex, (i - 1, j + 1)), 2),
                        ((vertex, (i - 1, j)), 2)]
            elif j == 0:
                return [((vertex, (i, j + 1)), 1),
                        ((vertex, (i - 1, j + 1)), 2),
                        ((vertex, (i - 1, j)), 2)]
            elif j == n - 1:
                return [((vertex, (i, j - 1)), 2),
                        ((vertex, (i - 1, j)), 4)]
                
        # Middle vertices:
        elif i in range(1, n - 1):
            if i % 2 == 1: # Even row (i starts at 0)
                if j in range(1, n - 1):
                    return [((vertex, (i, j - 1)), 1),
                            ((vertex, (i + 1, j - 1)), 1),
                            ((vertex, (i + 1, j)), 1),
                            ((vertex, (i, j + 1)), 1),
                            ((vertex, (i - 1, j)), 1),
                            ((vertex, (i - 1, j - 1)), 1)]
                elif j == 0:
                    return [((vertex, (i + 1, j)), 2),
                            ((vertex, (i, j + 1)), 2),
                            ((vertex, (i - 1, j)), 2)]
                elif j == n - 1:
                    return [((vertex, (i, j - 1)), 1),
                            ((vertex, (i + 1, j - 1)), 1),
                            ((vertex, (i + 1, j)), 1),
                            ((vertex, (i - 1, j)), 1),
                            ((vertex, (i - 1, j - 1)), 1)]
            elif i % 2 == 0: # Odd row
                if j in range(1, n - 1):
                    return [((vertex, (i, j - 1)), 1),
                            ((vertex, (i + 1, j)), 1),
                            ((vertex, (i + 1, j + 1)), 1),
                            ((vertex, (i, j + 1)), 1),
                            ((vertex, (i - 1, j + 1)), 1),
                            ((vertex, (i - 1, j)), 1)]
                elif j == n - 1:
                    return [((vertex, (i, j - 1)), 2),
                            ((vertex, (i + 1, j)), 2),
                            ((vertex, (i - 1, j)), 2)]
                elif j == 0:
                    return [((vertex, (i + 1, j)), 1),
                            ((vertex, (i + 1, j + 1)), 1),
                            ((vertex, (i, j + 1)), 1),
                            ((vertex, (i - 1, j + 1)), 1),
                            ((vertex, (i - 1, j)), 1)]
        # If none of the conditions are fulfilled, raise an error.
        raise ValueError('The vertex is not in the grid.') 
    elif cond == 'Dirichlet':
        raise NotImplementedError
    
    
# So far so good :)

def A(n : int, cond='Neumann'):
    '''Returns the adjacency matrix of the grid.'''
    out = np.zeros((n ** 2, n ** 2), dtype=int)
    vertices = [(i, j) for i in range(n) for j in range(n)]
    if cond == 'Neumann':
        for vertex in vertices:
            # print('#================================#')
            # print('Vertex: ', vertex, '\n', 'Edges: ')
            for edge in edgeTemplate(vertex, n):
                # print(edge)
                out[vertices.index(vertex)][vertices.index(edge[0][1])] = edge[1]
    # print('#================================#')
    return out
    
def LaplacianSim(n : int, cond='Neumann'):
    '''Returns the simmetrised Laplacian matrix of the grid
    using the formula 
            L[i, j] = D[i, j] - sqrt(A[i, j] * A[j, i]) '''
    return D(n, cond) - np.sqrt(A(n, cond) * A(n, cond).T)


def B(n : int, cond='Neumann'):
    '''Returns the factorisation of L into the product
    of two matrices, B and B^H.'''
    vertices = [(i, j) for i in range(n) for j in range(n)]
    adjM = A(n, cond)
    edges = []
    if cond == 'Neumann':
        out = np.zeros((n ** 2, 3 * n ** 2 - 4 * n + 1), dtype=float)
    elif cond == 'Dirichlet':
        raise NotImplementedError
    
    # Getting the list of all directed edges
    for vertex in vertices:
        for edge in edgeTemplate(vertex, n):
            edges.append(edge)
            
    # Getting the list of all undirected edges 
    edgesUndirected = []
    for e in edges:
        if e in edgesUndirected or (e[0][1], e[0][0]) in edgesUndirected:
            continue
        else:
            edgesUndirected.append((e[0][0], e[0][1]))
            
    # Filling the matrix B
    for k, e in enumerate(edgesUndirected):
        print('#================================#')
        # print('k: ', k)
        (i, j) = e
        # We have to transform i and j into the corresponding index in the vertices list
        i, j = vertices.index(i), vertices.index(j)
        print('Edge: ', e)
        i, j = min(i, j), max(i, j)
        print('Vertices: ', i, ',', j)
        if i == j:
            out[i][k] = np.sqrt(adjM[i][j])
            print('Diagonal: ', adjM[i][j])
        else:
            out[i][k] = np.sqrt(adjM[i][j])
            out[j][k] = -np.sqrt(adjM[j][i])
            print('Off-diagonal: ', adjM[i][j], adjM[j][i])
    print('#================================#')
    return out        
        
def BHamiltonian2D_H(n : int, B, cond='Neumann'):
    '''Given an integer [n] and matrix [B], the function returns the Hamiltonian matrix 
    for a graph with [n ** 2] vertices and [2 * n ** 2 - 2 * n] edges, as dictated by the article.'''
    if cond == 'Neumann':
        H = np.block([[np.zeros((n ** 2, n ** 2)), B], [np.transpose(B), np.zeros((3 * n ** 2 - 4 * n + 1, 3 * n ** 2 - 4 * n + 1))]]) * np.sqrt(2 / 3) / a 
    elif cond == 'Dirichlet':
        # H = np.block([[np.zeros((n ** 2, n ** 2)), B], [np.transpose(B), np.zeros((2 * n ** 2 + 2 * n - 4, 2 * n ** 2 + 2 * n - 4))]]) / (n + 1)
        raise NotImplementedError
    return H

# print(A(3), '\n')
# print(B(3), '\n')
# print(LaplacianSim(3), '\n')






#==============================================================================#
#                             INITIAL CONDITIONS                               #
#------------------------------------------------------------------------------#

# We wish to use the Gaussian pulse as an initial condition for now. Later 
# on, we will also implement the Richer wavelet.
#
# The first step is to define the hexagonal grid:

def hexGridEuclidean(n : int, a : float, cond='Neumann'):
    '''Returns a grid of [n] ** 2 points in the Euclidean plane. We imagine the
    grid as a hexagonal lattice with spacing between the vertices being [a].'''
    out = []
    if cond == 'Neumann':
        l = [a/2 + i * a for i in range(n)]
        for i in range(n):
            if i % 2 == 0:
                out.append([(a/2 + j * a, a * np.sqrt(3) * i / 2) for j in range(n)])
            else:
                out.append([(j * a, a * np.sqrt(3) * (i) / 2) for j in range(n)])
        out.reverse()
    elif cond == 'Dirichlet':
        raise NotImplementedError
    return np.array(out)



# x = []
# y = []
# for i in hexGridEuclidean(3, 1):
#     for j in i:
#         x.append(j[0])
#         y.append(j[1])
# fig = plt.figure()
# plt.scatter(x, y)
# plt.show()



def initialGaussian(n : int, a : float, var, cond='Neumann'):
    '''Returns the initial condition of the Gaussian pulse on the hexagonal grid.
    The pulse is centered at the central point in the grid.'''
    out = []
    grid = hexGridEuclidean(n, a, cond)
    mux = grid[n // 2][n // 2][0]
    muy = grid[n // 2][n // 2][1]
    for i, row in enumerate((grid)):
        out.append([])
        for vert in row:
            print(vert)
            print(HE.gaussian2D(vert[0], vert[1], mux, muy, var))
            out[i].append(HE.gaussian2D(vert[0], vert[1], mux, muy, var))
            print(colored(out, 'light_green'), end='\n\n')
    return np.array(out)
            
            
def initialFix(init, cond='Neumann'):
    '''Given the initial condition state, the function returns a corrected 
    initial state taking into account the correction factors for the 
    boundary vertices.'''
    if cond == 'Neumann':
        # The correction factors are 1/sqrt2 for the vertices on the boundary
        for i in range(len(init)):
            for j in range(len(init[i])):
                if i in [0, len(init) - 1] or j in [0, len(init[i]) - 1]:
                    init[i][j] /= np.sqrt(2)
    elif cond == 'Dirichlet':
        raise NotImplementedError 
    return init

def initialFix2(init, cond='Neumann'):
    '''Given the corrected initial state, the function adds the values 
    corresponding to the edge-values of the electric field. In the simplest 
    case it adds zeros and vectorizes the grid.'''
    n = len(init)
    vec = init.flatten()
    if cond == 'Neumann':
        out = np.append(vec, np.zeros(3 * n ** 2 - 4 * n + 1))
    elif cond == 'Dirichlet':
        raise NotImplementedError
    return out

# Example:
# init = initialGaussian(3, 1, 1)
# print(init, '\n')
# init2 = initialFix(init)
# print(init2, '\n')
# init3 = initialFix2(init2)
# print(init3, '\n')


        
#==============================================================================#
#                             MAIN FUNCTION                                    #
#------------------------------------------------------------------------------#

def main(n : int, a : float, var, t : int, dt : int, cond='Neumann'):
    '''The main function that evolves the initial state according to the 
    Hamiltonian dynamics.'''
    # Initial condition:
    init = initialFix2(initialFix(initialGaussian(n, a, var), cond), cond)
    
    # Hamiltonian matrices:
    L = LaplacianSim(n, cond)
    Bmat = B(n, cond)
    H = BHamiltonian2D_H(n, Bmat, cond)
    # Evolution:
    state = HE.evolve()
    return state


def animateEvolution2D(H, psi0, tmax, dt, n : int):
    '''Animates the field of each vertex under the influence of
    a Hamiltonian [H] and given the starting state [psi0]. The
    discretization is dictated by the time step [dt] and the
    total time [tmax].
    
    In this function we specify the number of vertices [n].'''
    # Data preparation
    m = len(psi0) # m is the size of the initial state
    ts = np.arange(0, tmax, dt) # Time steps

    # Evolution
    wavefunctions = []
    wavefunctions.append(np.sqrt(psi0.probabilities()[:n]))
    for i, t in enumerate(ts):
        psi = HE.evolveTime(H, t, psi0)
        vals = np.sqrt(psi.probabilities()[:n])
        wavefunctions.append(vals)
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(colored('Evolutions completed. Plotting...', 'green'))
           
    # Plotting
    global wave
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))    
    x = [i / (n-1) for i in range(a)]
    y = [i / (n-1) for i in range(a)]
    x, y = np.meshgrid(x, y)
    
    # Convert the wavefunctions to a 2D array
    wavefunctionsFIXED = []
    for wave in wavefunctions:
        wave = np.array(wave).reshape((a, a))
        wavefunctionsFIXED.append(wave)
        
    
    wave = [ax.plot_surface(x, y, wavefunctionsFIXED[0], color='b')]
    #ax.set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Amplitude')
        
    def update(frame, wave, wavefunctionsFIXED):
        z = wavefunctionsFIXED[frame]
        #print(wave)
        wave[0].remove()
        wave[0] = ax.plot_surface(x, y, z, color='b')
        return wave
    
    ax.set_zlim(0, 0.5)
    anime = animation.FuncAnimation(fig=fig, func=update, fargs=(wave, wavefunctionsFIXED),frames=(len(ts) - 1), interval=1000 // fps)
    plt.show()


# TODO:
# 1. Implement the function above in a correct manner to accomodate the hexagonal grid.
# 2. Test the function with a simple example.
# 3. Examine the results and see if they are correct.
# 4. Expand the function to include the Dirichlet boundary conditions.
# 5. (Optional) Implement the Richer wavelet as an initial condition.
# 6. (Optional) Implement the function to animate the evolution of the wavelet.
# 7. (Long term) Implement the Dirichlet boundary conditions.