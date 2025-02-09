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