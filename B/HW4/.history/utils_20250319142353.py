import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla

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


# def newton_krylov(U_0, h, N, b2, r, tol, max_iter):
#     U = U_0.copy()
    
#     for i in range(max_iter):
#         F_U = F(U, h, N, b2, r)  # Compute function value
        
#         norm_F = np.linalg.norm(F_U, ord=np.inf)
#         print(f"Iteration {i}, ||F(U)|| = {norm_F}")
        
#         if norm_F < tol:
#             print("Converged!")
#             return U
        
#         J_U = J(U, h, N, b2, r)  # Compute Jacobian
#         dU, _ = spla.gmres(J_U, -F_U, tol=tol)  # Solve J(U) dU = -F(U) using GMRES
        
#         U += dU  # Update solution
        
#     raise RuntimeError("Newton-Krylov did not converge within the given iterations")
