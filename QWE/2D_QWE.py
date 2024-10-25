import numpy as np
from termcolor import colored
from Hamiltonians.Libraries import HamiltonianEvolution as HE
import matplotlib.pyplot as plt
import matplotlib as mpl


#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n = 21
var = 1.5
t = 1000
dt = 10
a = 0.5

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
        H = HE.BHamiltonian2D(n, B_Neumann(n), a, 'Neumann')
    else:
        H = HE.BHamiltonian2D(n, B_Dirichlet(n), a, 'Dirichlet')

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
    
    
# simulation2D('Dirichlet')





#==============================================================================#
#                           PLOTTING FUNCTIONS                                 #
#------------------------------------------------------------------------------#

def gridEuclidean(n : int, a : float):
    '''Returns a square grid of n ** 2 points in the Euclidean plane.'''
    out = n * []
    for i in range(n):
        out.append(n * [])
        for j in range(n):
            out[i].append((i * a, j * a))
    return out



def initialRicher(n : int, a : float, var, cond='Neumann'):
    '''Returns the initial condition of the Richer wavelet on the hexagonal grid.
    The pulse is centered at the central point in the grid.'''
    out = []
    grid = gridEuclidean(n, a)
    mux = grid[n // 2][n // 2][0]
    muy = grid[n // 2][n // 2][1]
    for i, row in enumerate((grid)):
        out.append([])
        for vert in row:
            out[i].append(HE.richer2D(vert[0], vert[1], mux, muy, var))
    out = np.array(out).flatten()
    if cond == 'Neumann':
        out2 = np.concatenate((out, np.zeros(2 * n ** 2 - 2 * n)), axis=0)
    elif cond == 'Dirichlet':
        raise NotImplementedError
    return np.array(out2)


def plotWavefunction2D(H, psi0, t, n : int):
    '''Plots the wavefunction given by the Hamiltonian H 
    at time t of the initial condition psi0.
    The integer n ** 2 is the number of vertices.'''
    print(colored('Evolving...', 'blue'), end='\r')
    psi = HE.evolveTime(H, t, psi0)
    print(psi)
    print(colored('Evolution completed.', 'green'))
    wavefunction = np.real(psi.data[:n**2])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    print(colored('Preparing grid...', 'blue'), end='\r')
    grid = gridEuclidean(n, a)
    print(colored('Grid prepared.    ', 'green'))
    xs = []
    ys = []
    for row in grid:
        for vert in row:
            xs.append(vert[0])
            ys.append(vert[1])
    print(colored('Data extracted.', 'green'))
    wave = ax.plot_trisurf(xs, ys, wavefunction, cmap=mpl.colormaps['viridis'])
    plt.show()
    return wavefunction

print(B_Neumann(n))
print(B_Neumann(n).shape, 'B')
H = HE.BHamiltonian2D(n, B_Neumann(n), a, 'Neumann')
print(H.shape, 'H')
init = initialRicher(n, a, var)
psi0 = HE.Statevector(init / HE.euclidean_norm(init))
print(psi0.data.size, 'psi0')
wave = plotWavefunction2D(H, psi0, 1.5, n)



#==============================================================================#
#                            ANISOTROPY ANALYSIS                               #
#------------------------------------------------------------------------------#

# To measure the anisotropy of our wave model, we are going to model the system 
# at a specific time t and measure distance between the source and the points
# in which the wave has peaked.

def peaks(wavefunction, n : int, eps : float):
    '''Returns the peaks of the wavefunction at a given time [t]. The peaks are 
    the points at which the wavefunction differs from the maximum by less than 
    the threshold [eps].'''
    M = max(wavefunction)
    out = []
    for i, val in enumerate(wavefunction):
        if M - val < eps:
            out.append(i)
    return out


def anisotropy(wavefunction, n : int, eps : float):
    '''Returns the anisotropy of the wavefunction at a given time [t]. The 
    anisotropy is the ratio of the distance between the source and the 
    farthest peak to the distance between the source and the nearest peak.'''
    grid = gridEuclidean(n, a)
    peaksList = peaks(wavefunction, n, eps)

    source = grid[n // 2][n // 2]
    print(source)
    maxDist = 0
    minDist = 10 ** 10
    for peak in peaksList:
        # print(peak)
        dist = np.sqrt((source[0] - grid[peak // n][peak % n][0]) ** 2 + (source[1] - grid[peak // n][peak % n][1]) ** 2)
        if dist > maxDist:
            maxDist = dist
            maxPeak = peak
        if dist < minDist:
            minDist = dist
            minPeak = peak
    print('#================================#')
    print('Max peak: ', maxPeak)
    print('Max distance: ', maxDist)
    theta1 = np.arctan((grid[maxPeak // n][maxPeak % n][1] - source[1]) / (grid[maxPeak // n][maxPeak % n][0] - source[0]))
    print('Theta1: ', theta1)
    print('#--------------------------------#')
    print('Min peak: ', minPeak)
    print('Min distance: ', minDist)
    theta2 = np.arctan((grid[minPeak // n][minPeak % n][1] - source[1]) / (grid[minPeak // n][minPeak % n][0] - source[0]))
    print('Theta2: ', theta2)
    print('#--------------------------------#')
    print('Anisotropy: ', maxDist / minDist)
    print('#================================#')
    return maxDist / minDist

print(anisotropy(wave, n, 0.01))

# TODO: For some reason the time evolution is affected by the precision of the grid.