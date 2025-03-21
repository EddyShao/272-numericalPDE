import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *

# Solve the system using Newton-Krylov



#### SH23 parameters ####
L = 20*np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter
r = -0.0 # bifurcation parameter

#### Newton-Krylov parameters ####
TOL = 1e-10
MAX_ITER = 1000

#### Initial condition ####
beta = 0.1 # beta is the scale of the initial u
u_0 = beta * np.sin(2 * np.pi * x / L)
v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u


U_0 = np.concatenate((u_0, v_0))

# compute eigenvalues
Jacobian = J(U_0, h, N, b2, r)


J11 = Jacobian[:N, :N]
J12 = Jacobian[:N, N:]
J21 = Jacobian[N:, :N]
J22 = Jacobian[N:, N:]








