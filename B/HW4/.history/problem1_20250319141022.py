import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt



L = 20*np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter

r = -1.0 # bifurcation parameter

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
    J[:N, :N] = (A + M)
    J[:N, N:] = -M
    J[N:, :N] =  M @ diags((r + 2 * b2 * u - 3 * u**2).flatten(), offsets=0, format="csr")
    J[N:, N:] =  -  (A + M)
  
    return J.tocsr()


def newton_krylov(U_0, h, N, b2, r, tol, max_iter):
    U = U_0.copy()
    
    for i in range(max_iter):
        F_U = F(U, h, N, b2, r)  # Compute function value
        
        norm_F = np.linalg.norm(F_U, ord=np.inf)
        print(f"Iteration {i}, ||F(U)|| = {norm_F}")
        
        if norm_F < tol:
            print("Converged!")
            return U
        
        J_U = J(U, h, N, b2, r)  # Compute Jacobian
        dU, _ = spla.gmres(J_U, -F_U, tol=tol)  # Solve J(U) dU = -F(U) using GMRES
        
        U += dU  # Update solution
        
    raise RuntimeError("Newton-Krylov did not converge within the given iterations")



# test

TOL = 1e-10
MAX_ITER = 1000

# initial guess
beta = 0.1 # beta is the scale of the initial u
u_0 = beta * np.sin(2 * np.pi * x / L)
v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u


U_0 = np.concatenate((u_0, v_0))
U = newton_krylov(U_0, h, N, b2, r, tol=TOL, max_iter=MAX_ITER)
u = U[:N]

# truncate u with machine epsilon
u[np.abs(u) < np.finfo(float).eps] = 0.0

plt.plot(x, u)  
plt.show()


