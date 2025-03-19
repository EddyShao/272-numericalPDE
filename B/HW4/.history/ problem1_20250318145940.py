import numpy as np
from scipy.sparse import spdiags, lil_matrix
import matplotlib.pyplot as plt



L =  1.
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 100)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter
beta = 0.1
u_0 = beta * np.sin(np.pi * x / L)
v_0 = - (np.pi/L)**2 u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u
r = 1.0 # bifurcation parameter
U_0 = np.concatenate((u_0, v_0))
TOL = 1e-6
MAX_ITER = 1000

def assemble_mass(N, h):
    e = np.ones(N)  # Vector of ones
    diagonals = [e, 4 * e, e]
    offsets = [-1, 0, 1]

    # Construct the main tridiagonal part
    M = (h / 6) * spdiags(diagonals, offsets, N, N, format="lil")

    # Add periodic boundary conditions
    M[0, N-1] = h / 6
    M[N-1, 0] = h / 6

    return M.tocsr()  # Convert to efficient sparse format for computations

def assemble_stiffness(N, h):
    e = np.ones(N)  # Vector of ones
    diagonals = [-e, 2 * e, -e]
    offsets = [-1, 0, 1]

    # Construct the main tridiagonal part
    A = (1 / h) * spdiags(diagonals, offsets, N, N, format="lil")

    # Add periodic boundary conditions
    A[0, N-1] = - 1 / h
    A[N-1, 0] = - 1 / h

    return A.tocsr()  # Convert to efficient sparse format for computations

def F(U, h, N, b2, r):
    u = U[:N].reshape(-1, 1)
    v = U[N:].reshape(-1, 1)
    A = assemble_stiffness(N, h)
    M = assemble_mass(N, h)
    F = np.zeros(2 * N)
    F[:N] = ((A + M) @ u - M @ v).flatten()
    F[N:] = (M@(r*u + b2 * u**2 - u**3) - (A + M) @ v).flatten()  
    return F

def J(U, h, N, b2, r):
    u = U[:N].reshape(-1, 1)
    v = U[N:].reshape(-1, 1)
    A = assemble_stiffness(N, h)
    M = assemble_mass(N, h)
    J = lil_matrix((2 * N, 2 * N))
    J[:N, :N] = A + M
    J[:N, N:] = - M
    J[N:, :N] = M @ np.diag(r + 2 * b2 * u - 3 * u**2)
    J[N:, N:] = - (A + M)
    return J.tocsr()

print("F(U_0) = ", F(U_0, h, N, b2, r))
print("J(U_0) = ", J(U_0, h, N, b2, r))