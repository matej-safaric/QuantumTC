import numpy as np

def grid(n : int):
    '''Returns a square grid of n ** 2 points.'''
    out = n * []
    for i in range(n):
        out.append(n * [])
        for j in range(n):
            out[i].append((i, j))
    return out


def edges(grid):
    '''Returns the edges of a grid.'''
    n = len(grid)
    out = []
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                out.append((grid[i][j], grid[i + 1][j]))
            if j < n - 1:
                out.append((grid[i][j], grid[i][j + 1]))
    return out

def conc(grid):
    '''Returns the concatenation of the points in a grid.'''
    out = []
    for row in grid:
        for point in row:
            out.append(point)
    return out

def B(n : int):
    '''Returns the B matrix for a grid of n ** 2 points.'''
    g = grid(n)
    e = edges(g)
    g = conc(g)
    out = np.zeros((len(e), len(g)))
    for i, (a, b) in enumerate(e):
        out[i][g.index(a)] = 1
        out[i][g.index(b)] = -1
    return out.transpose()

print(B(2))

# TODO: Sort out the order of the points in the grid.

