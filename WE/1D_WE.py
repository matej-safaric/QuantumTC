import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from typing import Callable as func

#==============================================================================#
#                                   DESCRIPTION                                #
#------------------------------------------------------------------------------#

# The purpose of this code is to create a simple simulation of the solution
# to the 1-dimensional wave equation, i.e. model a simple function of the 
# form f(x - ct)


#==============================================================================#
#                                 GLOBAL VARIABLES                             #
#------------------------------------------------------------------------------#

n = 30 # Discretization


#==============================================================================#
#                                     MAIN                                     #
#------------------------------------------------------------------------------#

def richer(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * np.sin(2 * np.pi * x)


def plotWave(initF : func, t : float):
    w = []
    xs = np.arange(0, 1, 1/n)
    print(xs)
    for x in xs:
        w.append(initF(x - t))
    print(w)
    plt.plot(xs, w, color='blue')
    plt.show()
    
plotWave(lambda x : richer(x, 0.5, 0.2), 0.2)