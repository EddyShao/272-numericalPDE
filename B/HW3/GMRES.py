import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def gmres(A_mv, b, x=None, m=None, threshold=1e-8):
    """
    Solve Ax = b using GMRES method.
    Input:
        A_mv: a function that takes x and returns Ax
        b: the right-hand side of the equation
        x: the initial guess of the solution
        m: the maximum number of iterations
        threshold: the threshold of the relative residual norm
    Output:
        x: the solution
        e: the relative residual norm at each iteration
    
    Refs: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
    """
    n = len(b)
    if x is None:
        x = np.zeros(n)
    if m is None:
        m = n

    r = b - A_mv(x)
    b_norm = np.linalg.norm(b)
    error = np.linalg.norm(r) / b_norm
    
    sn = np.zeros(m)
    cs = np.zeros(m)
    e = [error]
    
    r_norm = np.linalg.norm(r)
    Q = np.zeros((n, m+1))
    Q[:, 0] = r / r_norm
    
    beta = np.zeros(m+1)
    beta[0] = r_norm
    
    H = np.zeros((m+1, m))
    
    for k in range(m):
        H[:k+2, k], Q[:, k+1] = arnoldi(A_mv, Q, k)
        H[:k+2, k], cs[k], sn[k] = apply_givens_rotation(H[:k+2, k], cs, sn, k)
        
        beta[k + 1] = -sn[k] * beta[k]
        beta[k] = cs[k] * beta[k]
        error = abs(beta[k + 1]) / b_norm
        e.append(error)
        
        if error <= threshold:
            break
    
    y = np.linalg.solve(H[:k+1, :k+1], beta[:k+1])
    x += Q[:, :k+1] @ y
    
    return x, np.array(e)

def arnoldi(A_mv, Q, k):
    q = A_mv(Q[:, k])
    h = np.zeros(k+2)
    
    for i in range(k+1):
        h[i] = np.dot(q, Q[:, i])
        q -= h[i] * Q[:, i]
    
    h[k+1] = np.linalg.norm(q)
    if h[k+1] > 0:
        q /= h[k+1]
    
    return h, q

def apply_givens_rotation(h, cs, sn, k):
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i] = temp
    
    cs_k, sn_k = givens_rotation(h[k], h[k+1])
    h[k] = cs_k * h[k] + sn_k * h[k+1]
    h[k+1] = 0.0
    
    return h, cs_k, sn_k

def givens_rotation(v1, v2):
    t = np.sqrt(v1**2 + v2**2)
    cs = v1 / t
    sn = v2 / t
    return cs, sn


def jacobian_vector_product(v, u, h, r):
    """
    Computes the Jacobian-vector product J(v) for the Swift-Hohenberg equation:
    Parameters:
    - v  : np.array, perturbation vector (same shape as u)
    - u  : np.array, field variable at current step
    - dx : float, grid spacing
    - r  : float, bifurcation parameter

    Returns:
    - Jv : np.array, result of Jacobian-vector product
    """
    N = len(u)
    Jv = np.zeros_like(u)

    # Finite difference coefficients

    # Compute Laplacian (∂_x^2 v)
    laplacian = np.zeros_like(u)

    for i in range(1, N-1):
        laplacian[i] = (v[i-1] - 2*v[i] + v[i+1]) / h**2

    # Apply boundary conditions (Neumann, second derivative = 0)
    # laplacian[0] = (v[1] - v[0]) / h**2
    # laplacian[N-1] = (v[N-2] - v[N-1]) / h**2
    laplacian[0] = 0
    laplacian[N-1] = 0

    # Compute (1 + ∂_x^2) v
    op_v = v + laplacian

    # Compute (1 + ∂_x^2)^2 v = (1 + ∂_x^2) (op_v)
    laplacian2 = np.zeros_like(u)

    for i in range(1, N-1):
        laplacian2[i] = (op_v[i-1] - 2*op_v[i] + op_v[i+1]) / h**2

    # laplacian2[0] = (op_v[1] - op_v[0]) / h**2
    # laplacian2[N-1] = (op_v[N-2] - op_v[N-1]) / h**2
    laplacian2[0] = 0
    laplacian2[N-1] = 0
    

    op2_v = op_v + laplacian2

    # Compute Jacobian-vector product
    Jv = r * v - op2_v - 3 * (u**2) * v

    return Jv


if __name__ == '__main__':
    N = 301
    L = 20 * np.pi
    h = L / (N - 1)
    r_space = np.linspace(-0.5, 0.5, 5)
    # Initial field and perturbation vector
    u = np.zeros(N)  

    residuals = []



    b = np.random.randn(N)

    for r in r_space:
        # Initial guess
        A_mv = partial(jacobian_vector_product, u=u, h=h, r=r)
        v, e = gmres(A_mv, b)
        residuals.append(e)

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, residual in enumerate(residuals):
        ax[i].plot(residual, label=f'r = {r_space[i]}')
        ax[i].legend()
    plt.suptitle(r'Conevergence of GMRES')
    plt.savefig('GMRES.png')
    plt.show()