import numpy as np

def grid(n : int):
    '''Returns a square grid of n ** 2 points.'''
    out = n * []
    for i in range(n):
        out.append(n * [])
        for j in range(n):
            out[i].append((i, j))
    return np.array(out)

