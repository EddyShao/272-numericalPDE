
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def newton_krylov_solver(u0, h, r, F_u, tol=1e-4, max_iter=200):
    """
    Solve the Swift-Hohenberg equation using a Newton-Krylov solver.
    
    Parameters:
        u0 (numpy.ndarray): Initial guess of the solution.
        dx (float): Grid spacing.
        r (float): Bifurcation parameter.
        tol (float): Tolerance for the residual norm.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        numpy.ndarray: Solution of the Swift-Hohenberg equation.
    """
    u = u0.copy()
    N = len(u)
    F = partial(F_u, r=r, h=h)
    alpha = 1.0
    b = - F(u)
    error_hist = [np.linalg.norm(b)]

    for i in range(max_iter):
        A_mv = partial(jacobian_vector_product, u=u, h=h, r=r)
        v, _ = gmres(A_mv, b)
        
        line_search_flag = False

        for line_search_iter in range(100):
            u_new = u + alpha * v
            b = - F(u_new)
            error = np.linalg.norm(b)
            if error < error_hist[-1] - tol*0.01:
                line_search_flag = True
                break
            alpha *= 0.50
        
        if not line_search_flag:
            # raise warning
            print('WARNING: Line search failed')
            return u, error_hist
        

        # while np.linalg.norm(F(u + alpha * v)) >  error_hist[-1] - 1e-6:
        #     alpha *= 0.5
        u +=  alpha * v
        
        b = - F(u)
        error = np.linalg.norm(b)
        error_hist.append(error)
        print(f'Iteration {i+1}, Residual = {error:.3e}')
        if error < tol:
            break
    
    return u, error_hist


if __name__ == "__main__":
    from JFNK import newton_krylov_solver
    from ResidualF import F_u

    L = 20 * np.pi  # Domain length
    N = 301  # Number of grid points
    h = L / (N - 1)  # Grid spacing

    x_space = np.linspace(0, L, N)  # Grid points

    u0 = np.sin(2*np.pi*x_space / L)  # Initial guess
    r = -0.2
    u, error_hist = newton_krylov_solver(u0, h, r, F_u)
    
