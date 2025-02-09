def P_help(N, h):
    """
    Constructs the tridiagonal matrix P = (1 - ∂_x^2) in sparse format.
    P is used for the preconditioner P^2.
    """
    diag = np.ones(N) + 2/h**2  # Main diagonal
    off_diag = -1/h**2 * np.ones(N-1)  # Off-diagonal

    P = sp.diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], format='csc')
    return P

def P_inv2(P, b):
    """
    Solves the linear system P^2 x = b
    """
    P_factorized = spla.splu(P)
    y = P_factorized.solve(b)   # Solve P y = b
    x = P_factorized.solve(y)   # Solve P x = y
    return x


# Let M = p^{2}
# J M^{-1} M \delta u = -F(u)
# preconditioned Jacobian-vector product
# The idea is to solve P^2 v = Jv

def newton_krylov_solver_preconditioned(u0, h, r, tol=1e-4, max_iter=200, M_inv=None):
    """
    Solve the Swift-Hohenberg equation using a Newton-Krylov solver.
    
    Parameters:
        u0 (numpy.ndarray): Initial guess of the solution.
        dx (float): Grid spacing.
        r (float): Bifurcation parameter.
        tol (float): Tolerance for the residual norm.
        max_iter (int): Maximum number of iterations.
        M_inv (callable): Function that computes the inverse of the preconditioner.
    
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
        A_mv_pre = partial(jacobian_vector_product, u=u, h=h, r=r)
        A_mv = lambda v: A_mv_pre(M_inv(v))
        Mv, _ = gmres(A_mv, b)
        v = M_inv(Mv)
        
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
    
    return u, error_hist, 