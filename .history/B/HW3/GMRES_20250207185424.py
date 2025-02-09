import numpy as np

def gmres(matrix_vector_product, b, x0=None, tol=1e-6, max_iter=None, restart=None):
    """
    Matrix-free GMRES solver using Arnoldi with modified Gram-Schmidt.

    Parameters:
    - matrix_vector_product: Function that computes A @ v for any vector v.
    - b: Right-hand side vector.
    - x0: Initial guess (default: zero vector).
    - tol: Convergence tolerance.
    - max_iter: Maximum number of iterations (default: len(b)).
    - restart: Restarting frequency (default: no restart).

    Returns:
    - x: Solution vector.
    - residuals: List of residual norms at each iteration.
    """
    n = len(b)
    if max_iter is None:
        max_iter = n
    if restart is None:
        restart = max_iter
    if x0 is None:
        x0 = np.zeros(n)

    x = x0
    residuals = []

    for _ in range(max_iter // restart):
        # Compute initial residual
        r = b - matrix_vector_product(x)
        beta = np.linalg.norm(r)
        if beta < tol:
            residuals.append(beta)
            break
        
        # Arnoldi process
        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))
        V[:, 0] = r / beta
        
        for j in range(restart):
            w = matrix_vector_product(V[:, j])
            for i in range(j + 1):  # Modified Gram-Schmidt
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-12:  # Breakdown check
                break
            V[:, j + 1] = w / H[j + 1, j]

        # Solve least squares problem using QR decomposition
        e1 = np.zeros(restart + 1)
        e1[0] = beta
        Q, R = np.linalg.qr(H)
        y = np.linalg.solve(R[:restart, :], Q.T @ e1[:restart])

        # Update solution
        x += V[:, :-1] @ y
        res_norm = np.linalg.norm(matrix_vector_product(x) - b)
        residuals.append(res_norm)

        if res_norm < tol:
            break

    return x, residuals