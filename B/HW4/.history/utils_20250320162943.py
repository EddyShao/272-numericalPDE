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

def F_res(U, h, N, b2, r):
    u = U[:N]
    v = U[N:]
    A = assemble_stiffness(N, h)
    M = assemble_mass(N, h)
    f1 = (M + A).dot(u) - M.dot(v)
    f2 = -M.dot(r*u + b2*u**2 - u**3) + (M + A).dot(v)  
    return np.hstack([f1, f2])

def jacobian(U, h, N, b2, r):
    """Construct the Jacobian matrix (sparse) for FEM."""
    e = np.ones(N)
    A = sp.diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N)) * (1/h)
    A = A.tolil()
    A[N-1, 0] = -1/h
    A[0, N-1] = -1/h
    A = A.tocsr()

    M = sp.diags([e, 4*e, e], [-1, 0, 1], shape=(N, N)) * (h/6)
    M = M.tolil()
    M[N-1, 0] = h/6
    M[0, N-1] = h/6
    M = M.tocsr()

    u = U[:N]
    # Blocks
    J11 = M + A
    J12 = -M
    diag_term = r + 2*b2*u - 3*u**2
    J21 = -M.dot(sp.diags(diag_term, 0, shape=(N, N)))
    J22 = M + A

    top = sp.hstack([J11, J12])
    bottom = sp.hstack([J21, J22])
    J = sp.vstack([top, bottom])
    return J.tocsr()

def newton_krylov(U_0, h, N, b2, r, tol, max_iter):
    U = U_0.copy()
    
    for i in range(max_iter):
        F_U = F_res(U, h, N, b2, r)  # Compute function value
        
        norm_F = np.linalg.norm(F_U, ord=np.inf)
        print(f"Iteration {i}, ||F(U)|| = {norm_F}")
        
        if norm_F < tol:
            print("Converged!")
            return U
        
        J_U = J(U, h, N, b2, r)  # Compute Jacobian
        dU, _ = spla.gmres(J_U, -F_U, tol=tol)  # Solve J(U) dU = -F(U) using GMRES
        
        U += dU  # Update solution
        
    raise RuntimeError("Newton-Krylov did not converge within the given iterations")