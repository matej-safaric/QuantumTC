import numpy as np
from termcolor import colored
from Libraries import HamiltonianEvolution as HE
from scipy.linalg import expm, logm
from typing import Callable as func
import matplotlib.pyplot as plt
import time

#==============================================================================#
#                                 DESCRIPTION                                  #
#------------------------------------------------------------------------------#

# The purpose of this code is to see, whether it is possible to find such a 
# Hamiltonian for a quantum system that would allow for a smooth transition
# between the state |0> and the state |1>.

#==============================================================================#
#                             GLOBAL VARIABLES                                 #
#------------------------------------------------------------------------------#

n : int = 1
t : int = 1
dt = 0.01


#==============================================================================#
#                        SMOOTH TRANSITION FUNCTIONS                           #
#------------------------------------------------------------------------------#

def f(t):
    '''This function is a smooth, yet nonanalytic (at t = 0) function.'''
    return np.exp(-1/t) if t > 0 else 0

def g(t):
    '''This function is the aforementioned smooth transition function.'''
    return f(t) / (f(t) + f(1-t))

#-------------------------------- Plotting ------------------------------------#

def plot_g(t1, t2, dt):
    '''This function plots the smooth transition function.'''
    ts = np.arange(t1, t2, dt)
    gs = []
    for t in ts:
        gs.append(g(t))
    plt.plot(ts, gs)
    plt.show()
    
# plot_g(-1, 2, 0.01)

#--------------------------- Converting into Matrix ---------------------------#

def U_f(f : func, t):
    return np.array([[f(t), -np.sqrt(1 - f(t) ** 2)], [np.sqrt(1 - f(t) ** 2), f(t)]])

def H(t):
    return 1j * logm(U_f(g, t)) / t

def matNorm(M):
    '''Returns a simple norm (dunno if it's a norm) of a matrix.
    It sums up the squares of all the elements.'''
    out = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
           out += abs(M[i,j]) ** 2
    return out 

Ht = 0 
ts = np.arange(0.25, t, dt)
steps = []
H11 = []
H12 = []
H21 = []
H22 = []

fig, axs = plt.subplots(ncols=2)
H0 = H(0.25)
for t in ts:
    step = matNorm(H(t) - H0)
    steps.append(step)
    print(H(t)[0,:], '     ', H(t)[1,:], f'      Step: {step},    t: {t}')
    H11.append(abs(H(t)[0,0]))
    print(abs(H(t)[0,0]))
    H12.append(abs(H(t)[0,1]))
    print(abs(H(t)[0,1]))
    H21.append(abs(H(t)[1,0]))
    print(abs(H(t)[1,0]))
    H22.append(abs(H(t)[1,1]))
    print(abs(H(t)[1,1]))
    #Ht = H(t)
    time.sleep(0.1)
    
axs[0].plot(ts, steps)
axs[1].plot(ts, H11, label='H11')
axs[1].plot(ts, H12, label='H12')
axs[1].plot(ts, H21, label='H21')
axs[1].plot(ts, H22, label='H22')

plt.show()