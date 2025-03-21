import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import scipy.sparse as sp

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
    A = assemble_stiffness(N, h)
    M = assemble_mass(N, h)

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

def newton_krylov(F, U0, TOL, MAX_ITER, h, N, b2, r):
    """Solve F(U, r) = 0 using a Newton-Krylov method."""
    U = U0.copy()
    hist = []
    for k in range(MAX_ITER):
        Fx = F(U, r)
        res_norm = np.linalg.norm(Fx)
        hist.append(res_norm)
        if res_norm < TOL:
            break
        J = jacobian(U, h, N, b2, r)
        delta_U, info = spla.gmres(J, -Fx, tol=1e-6, restart=130)
        # As is suggested, we add a line search:
        alpha = 1.0
        flag = False
        for i in range(10):
            new_U = U + alpha * delta_U
            new_Fx = F(new_U, r)
            if np.linalg.norm(new_Fx) < res_norm:
                U = new_U
                flag = True
                break
            alpha *= 0.5
        if not flag:
            raise ValueError("Line search failed: could not find a better solution.")
            
    return U, np.array(hist)