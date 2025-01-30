import numpy as np
from scipy.linalg import expm
from termcolor import colored
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, Operator
from matplotlib import animation



#========================================================= DESCRIPTION =========================================================#
# The purpose of this code is to implement the Neumann boundary condition on a 2D discretization of the Euclidean plane
# when the normal to the boundary is the vector n = [1  1].
#===============================================================================================================================#


#-------- PREREQUISITES --------# 
#First some of the usual functions for modelling quantum systems:

def euclidean_norm(v):
    '''Given a vector v, the function returns the Euclidean norm of v.'''
    return np.sqrt(np.dot(v, np.conj(v)))

def evolutionTransform(H, t):
    return expm(-1j * H * t)

def evolve(H, psi0 : Statevector, T, dt):
    '''Returns the time evolution of the starting state [psi0] under the
    Hamiltonian H.'''
    evolution = []
    for i, t in enumerate(np.arange(0, T, dt)):
        U = evolutionTransform(H, t)
        U = Operator(evolutionTransform(H, t))
        psi = psi0.evolve(U)
        evolution.append(psi)
        print(colored(f'Evolution {i} out of {int(T/dt)} complete...', 'yellow'), end='\r')
    return evolution



# ...and the following aux function for turning a grid into a list
def conc(grid):
    '''Returns the concatenation of the points in a grid.'''
    out = []
    for row in grid:
        for point in row:
            out.append(point)
    return out













#-------- GRID SETUP --------#
# Here, we set up the matrices B and H that are required for the simulation

# To start with, define the variable n as the grid size (length of one side, assuming the grid is square)
n = 25

# Next, we need to know where the Neumann boundary lies. Since the directional vector that is parallel to the boundary 
# has the components [1  -1], it is adequate to present the index of the vertex on the leftmost vertical side of the 
# grid. From this index, we can calculate all the other indices on the boundary.
bi = 1 # <- (boundary index)




# Now it's time to set up the grid itself which is simple enough. We are only creating a list of all the vertices 
# in the standard reading order (left to right, top to bottom):
def grid(n):
    out = []
    for i in range(n):
        out.append([])
        for j in range(n):
            out[i].append((i, j))
    return out
# We will only be using this list as a reference so as to have an easier job with implementing the edge 
# weights.



# The next step is to add the list of all possible edges:
def edgeList(n):
    '''Returns the edges of a grid according to the Neumann
    boundary conditions.'''
    out = []
    for i in range(n):
        for j in range(n):
            if j < n - 1:
                out.append((grid[i][j], grid[i][j + 1]))
            if i < n - 1:
                out.append((grid[i][j], grid[i + 1][j]))
    return out


# Now for the edge weights in the graph. They will be immediately implemented via the matrix B.

def BMat(n : int):
    '''Returns the B matrix for a grid of n ** 2 points according to
    the Neumann boundary condition on the edge of the grid.'''
    g = grid(n)
    e = edgeList(n)
    g = conc(g)
    out = np.zeros((len(e), len(g)))
    for i, (a, b) in enumerate(e):
        out[i][g.index(a)] = 1
        out[i][g.index(b)] = -1
    return out.transpose()

# The function above only returns the edge weights for the case where the Neumann boundary is on the outer edges of the grid.
# We do not want this exact thing - we must first look at the discrete Laplacian and find out whether it can be decomposed 
# into a product of such a matrix B with its complex conjugate.






#-------------------- DISCRETE LAPLACIAN --------------------#
def reindexToConc(n, i, j):
    '''Returns the index of (i, j) in an nxn grid when mapped
    with conc().'''
    return n * i + j 

def reindexFromConc(n, i):
    '''If toConc = FALSE: Returns the index in the grid when 
    given the index of conc(grid).'''
    return (i // n, i % n)

def isvalid(n, i, j):
    return False if i not in range(n) or j not in range(n) else True

def Laplacian(n, bi):
    '''Returns the discrete Laplacian as described in the notebook.'''
    # We will be building the matrix row by row
    g = conc(grid(n))
    out = np.zeros((n**2, n**2))
    for pt in g:
        i, j = pt[0], pt[1]
        # Define the index of the element in the concatenated grid
        concInd = i * n + j
        if j == i - bi:
            # The vertices that lie on this line are on the diagonal Neumann boundary.
            # For these vertices, the expression for L is made up of three vertices:
            #       itself, its right neighbor, its upper neighbor
            out[concInd][concInd] = 4
            out[concInd][concInd + 1] = -2
            out[concInd][concInd - n] = -2
        elif j > i - bi:
            # Points above the boundary 
            def indices(i, j):
                '''Returns the list of valid indices around the point (i,j).'''
                tmpInd = i * n + j
                candidates = [tmpInd + 1, tmpInd - 1, tmpInd + n, tmpInd - n]
                return [k for k in candidates if isvalid(n, reindexFromConc(n, k)[0], reindexFromConc(n, k)[1])]
            
            for ind in indices(i,j):
                out[concInd][ind] = -1
            out[concInd][concInd] = 4
        else:
            continue      
    return out      





#------------------------ SIMMETRY ------------------------#
        
L = Laplacian(n, bi)
# print(L)

def isSymmetric(A, verbose : bool):
    """Checks if matrix is simmetric.
    *IMPORTANT:* Assumes that the matrix is an np.array!
    """
    if not verbose:
        return True if (A == A.T).all() else False
    else:
        outliers = []
        for i, row in enumerate(A):
            for j, col in enumerate(row):
                if A[i, j] != A[j, i]:
                    outliers.append((i, j))
        return (False, outliers) if outliers != [] else True


    
# print(isSymmetric(L, verbose=True))




#--------------------- NORMALITY -----------------------#

def isNormal(A):
    '''Checks if matrix is normal.'''
    l = A @ np.conj(A.T)
    d = np.conj(A.T) @ A
    if np.allclose(l, d):
        print('Normal: ', colored('TRUE', 'green'))
        return True
    else:
        print('Normal: ', colored('FALSE', 'red'))
        return False
    
    
# print(isNormal(L))

#--------------------- POSITIVE SEMI-DEFINITENESS ----------------------#
# It seems that the discrete Laplacian isn't simmetric.
# In order for there to be a square root of this matrix, we
# are hoping that it is positive semi-definite


# We are going to be checking this property via the matrix's leading 
# minors

from numpy.linalg import det

def isPosSD(A, verbose : bool):
    '''Checks whether a matrix is positive semi-definite.
    
    Parameters
    A : any matrix of the type np.array
    verbose : bool value that determines whether the function
    returns in depth information about the subdeterminants'''
    minors = []
    for i, _ in enumerate(A):
        d = det(A[:i, :i])
        minors.append(d)
    minors = np.array(minors)
    if verbose:
        return (minors[minors < 0])
    else:
        return 1
    
# print(isPosSD(L, verbose=True))


#=====================================================#
#   SUCCESS: This matrix is positive semi-definite!   #
#=====================================================#

# The consequence of this result is that there exists 
# a square root of this matrix.

from numpy.linalg import eig

# An idea for taking the square root of the matrix is 
# to diagonalize it first



#----------------------- DIAGONALIZATION ------------------------#

def diagonalize(A):
    '''Returns the pair (P, D) belonging to the diagonalization.'''
    eigval, eigvec = eig(A)
    return (eigvec, np.diag(eigval))

# (P, D) = diagonalize(L)

# The following print statement truly does return L
# print(np.real((P @ D @ np.linalg.inv(P)).round(0)))




#----------------------- SQUARE ROOT ----------------------------#

def matSqrt(A):
    '''Returns the square root of a diagonalizable matrix.'''
    (P0, D0) = diagonalize(A)
    D1 = np.sqrt(D0)
    return P0 @ D1 @ np.linalg.inv(P0)

# print(np.real(matSqrt(L).round(2)), '\n\n\n')
# print(np.real((matSqrt(L) @ matSqrt(L)).round(2)))














#------------------------ OTHER DECOMPOSITION ---------------------#

# Here we split L into the product of a matrix B with its complex 
# conjugate:
#                       L = BB*

# We can achieve this by taking the decomposition of L:
#                       L = QDQ*
# and taking 
#                       B = QD^0.5

#TODO: This may not work since the Laplacian is not necessarily simmetric








#===============================================================================#
#                                   PLOTTING                                    #
#===============================================================================#

# The following is a function, copied from HamiltonianEvolution.py, for 
# animating the 2D evolution of a quantum system:
def animateEvolution2D_V2(H, psi0, tmax, dt, n : int):
    '''Animates the field of each vertex under the influence of
    a Hamiltonian [H] and given the starting state [psi0]. The
    discretization is dictated by the time step [dt] and the
    total time [tmax].
    
    In this function we specify the number of vertices [n]. This function
    is adjusted to not include the edge values in the plot.'''
    
    # Time evolution function:
    def evolveTime(H, t, psi0 : Statevector):
        '''Given a Hamiltonian H, time t, and initial state psi0, return the 
        evolved state psi(t).
        
        The function also stores the unitary matrix U into a .json file.'''
        U = np.real(expm(-1j * H * t))
        psi_t = psi0.evolve(Operator(U))
        return psi_t






    # Data preparation

    a = int(np.sqrt(n)) # a is the number of vertices in one dimension
    ts = np.arange(0, tmax, dt) # Time steps



    # Evolution

    wavefunctionsR = []
    wavefunctionsI = []
    wavefunctionsR.append(np.real(psi0.data[:n]))
    wavefunctionsI.append(np.imag(psi0.data[:n]))
    for i, t in enumerate(ts):
        psi = evolveTime(H, t, psi0)
        # vals = np.sqrt(psi.probabilities()[:n])
        wavefunctionsR.append(np.real(psi.data[:n]))
        wavefunctionsI.append(np.imag(psi.data[:n]))
        print(f'Evolution {i} of {len(ts)} completed.', end='\r')
    print(colored('Evolutions completed. Plotting...', 'green'))


           

    # Plotting

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    #ax2 = fig.add_subplot(212, projection='3d')
    
    ax1.set_zlim(-0.5, 0.5)
    #ax2.set_zlim(-0.5, 0.5)
    
    x = [i / (a-1) for i in range(a)]
    y = [i / (a-1) for i in range(a)]
    
    x,y = np.meshgrid(x, y)


    
    # Convert the wavefunctions to a 2D array
    wavefunctionsR_FIXED, wavefunctionsI_FIXED = [], []
    for waveR in wavefunctionsR:
        wave = np.array(waveR).reshape((a, a))
        wavefunctionsR_FIXED.append(wave)
    for waveI in wavefunctionsI:
        wave = np.array(waveI).reshape((a, a))
        wavefunctionsI_FIXED.append(wave)

    waveR = [ax1.plot_surface(x, y, wavefunctionsR_FIXED[0])]
    #waveI = [ax2.plot_surface(x, y, wavefunctionsI_FIXED[0])]
    #axs[0].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Real Amplitude')
    #axs[1].set(xlim=[0, 1], ylim=[-1, 1], xlabel='Position', ylabel='Imaginary Amplitude')
        
        
        
    def update(frame, waveR, waveI, wavefunctionsR_FIXED, wavefunctionsI_FIXED):
        z1 = wavefunctionsR_FIXED[frame]
        waveR[0].remove()
        waveR[0] = ax1.plot_surface(x, y, z1, color='b')

        #z2 = wavefunctionsI_FIXED[frame]
        #waveI[0].remove()
        #waveI[0] = ax2.plot_surface(x, y, z2, color='r')
        return [waveR]
    
    anime = animation.FuncAnimation(fig=fig, func=update, fargs=(waveR, waveI, wavefunctionsR_FIXED, wavefunctionsI_FIXED), frames=(len(ts) - 1), interval=1000 // 30)
    plt.show()
   
   
# We also implement a function for plotting a singular point in time
def snapshot(H, psi0, t, n : int):
    '''Plots a single snapshot of the time evolution at time t.'''
    # First we add the mandatory function for time evolutions:
    # Time evolution function:
    def evolveTime(H, t, psi0 : Statevector):
        '''Given a Hamiltonian H, time t, and initial state psi0, return the 
        evolved state psi(t).
        
        The function also stores the unitary matrix U into a .json file.'''
        U = np.real(expm(-1j * H * t))
        psi_t = psi0.evolve(Operator(U))
        return psi_t
    
    sideLen = int(np.sqrt(n))
    psi = evolveTime(H, t, psi0)
    
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-0.5, 0.5)

    x = [i / (np.sqrt(n)-1) for i in range(sideLen)]
    y = [i / (np.sqrt(n)-1) for i in range(sideLen)]
    
    x,y = np.meshgrid(x, y)
    ax1.plot_surface(x, y, np.array(psi).reshape((sideLen, sideLen)))
    plt.show()

    
# Parameters:
tmax = 30
dt = 1

a = 1 / n # in order for the grid to always lie in [0, 1]x[0, 1]

mu = (0.5, 0.5)
var = 0.15

# Initial conditions:

# We make some changes to the previously used initialRicher() function:
def initialRicher(n : int, a : float, mu, var):
    '''Returns the initial condition of the Richer wavelet on the grid.
    The pulse is centered at [mu]. Unlike the previous versions, we 
    need not add the zeroes to the end of the list for the edge values.'''
    out = []
    
    def richer2D(x, y, mux, muy, sigma):
        '''Implements the richer wavelet in 2D aka the Mexican hat wavelet. This function is also the second derivative of the Gaussian function.'''
        return 1 / (np.pi * sigma ** 4) * (1 - ((x - mux) ** 2 + (y - muy) ** 2) / (2 * sigma ** 2)) * np.exp(-((x - mux) ** 2 + (y - muy) ** 2) / (2 * sigma ** 2))

    g = grid(n=n)
    
    def EuclideanGrid(grid, a : float):
        '''Given a grid of indices, the function returns a grid of points in R^2 with spacing [a].'''
        outTmp = []
        for row in grid:
            outTmp.append([(a * el[0], a * el[1]) for el in row])
        return np.array(outTmp)
    
    g = EuclideanGrid(grid = g, a = a)
    
    for row in g:
        out.append([richer2D(el[0], el[1], mu[0], mu[1], var) for el in row])        
    out = np.array(out)
    
    # Now we want to set the values to zero for the vertices that are 
    # outside the boundary:
    for i, _ in np.ndenumerate(out):
        if i[1] < i[0] - bi:
            out[i[0], i[1]] = 0
    out = out.flatten()
    return out / euclidean_norm(out)

init = Statevector(initialRicher(n, a, mu, var))
print(init.is_valid())
H = matSqrt(L)

animateEvolution2D_V2(H, init, tmax, dt, n ** 2)
# snapshot(H, init, 20, n ** 2)
