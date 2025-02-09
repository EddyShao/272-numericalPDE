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


