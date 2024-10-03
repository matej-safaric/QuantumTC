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
                            ((vertex, (i + 1, j - 1)), 1),
                            ((vertex, (i + 1, j)), 1),
                            ((vertex, (i, j + 1)), 1),
                            ((vertex, (i - 1, j)), 1),
                            ((vertex, (i - 1, j - 1)), 1)]
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
        out = np.zeros((n ** 2, 3 * n ** 2 - 4 * n + 1), dtype=int)
        for vertex in vertices:
            for edge in edgeTemplate(vertex, n):
                edges.append(edge)
        # Right now the edges list is too big for our purposes.
        # Let's count the amount of undirected edges (hence not
        # taking into account the direction of the edges)
        m = len(edges) // 2
    raise NotImplementedError
        
        

print(A(3), '\n')
print(B(3))