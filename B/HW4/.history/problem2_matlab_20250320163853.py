import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *



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

def compute_tangent(U, r, h, N, b2, tangent_prev):
    """Compute the tangent vector for pseudo-arclength continuation."""
    J = jacobian(U, h, N, b2, r)
    Fr = np.zeros(2*N)
    u = U[:N]
    e = np.ones(N)
    # Build M as in FEM
    M = sp.diags([e, 4*e, e], [-1, 0, 1], shape=(N, N)) * (h/6)
    M = M.tolil()
    M[N-1, 0] = h/6
    M[0, N-1] = h/6
    M = M.tocsr()
    Fr[:N] = -M.dot(u)
    dr_prev = tangent_prev[-1]
    if abs(dr_prev) < 1e-10:
        dr_prev = 1.0
    dU_temp = spla.spsolve(J, -Fr*dr_prev)
    tangent_temp = np.concatenate([dU_temp, [dr_prev]])
    # Augmented system: [J  Fr; tangent_temp^T] * tangent = [0; 1]
    J_dense = J.toarray()
    A_top = np.hstack([J_dense, Fr.reshape(-1, 1)])
    A_bottom = np.hstack([tangent_temp[:-1].reshape(1, -1), np.array([[tangent_temp[-1]]])])
    A = np.vstack([A_top, A_bottom])
    b_vec = np.concatenate([np.zeros(2*N), [1]])
    tangent = np.linalg.solve(A, b_vec)
    if np.any(np.isnan(tangent)) or np.any(np.isinf(tangent)):
        raise ValueError("Tangent computation failed: NaN or Inf values detected.")
    return tangent

def corrector(U_pred, r_pred, U_prev, r_prev, tangent_prev, ds, F, newton_tol, max_newton_iter, h, N, b2):
    """Perform the corrector (Newton) step for pseudo-arclength continuation."""
    U = U_pred.copy()
    r = r_pred
    for k in range(max_newton_iter):
        Fx = F(U, r)
        arc_res = np.dot(tangent_prev, np.concatenate([U, [r]]) - np.concatenate([U_prev, [r_prev]])) - ds
        residual = np.concatenate([Fx, [arc_res]])
        if np.linalg.norm(residual) < newton_tol:
            break
        J = jacobian(U, h, N, b2, r)
        Fr = np.zeros(U.size)
        u = U[:N]
        e = np.ones(N)
        M = sp.diags([e, 4*e, e], [-1, 0, 1], shape=(N, N)) * (h/6)
        M = M.tolil()
        M[N-1, 0] = h/6
        M[0, N-1] = h/6
        M = M.tocsr()
        Fr[:N] = -M.dot(u)
        J_dense = J.toarray()
        A_top = np.hstack([J_dense, Fr.reshape(-1, 1)])
        A_bottom = np.hstack([tangent_prev[:-1].reshape(1, -1), np.array([[tangent_prev[-1]]])])
        J_aug = np.vstack([A_top, A_bottom])
        delta = spla.gmres(sp.csr_matrix(J_aug), -residual, tol=1e-6, restart=130)[0]
        delta_U = delta[:-1]
        delta_r = delta[-1]
        alpha = 1.0
        for i in range(10):
            U_temp = U + alpha * delta_U
            r_temp = r + alpha * delta_r
            new_res = np.concatenate([F(U_temp, r_temp),
                                      [np.dot(tangent_prev, np.concatenate([U_temp, [r_temp]]) - np.concatenate([U_prev, [r_prev]])) - ds]])
            if np.linalg.norm(new_res) < np.linalg.norm(residual):
                U = U_temp
                r = r_temp
                break
            alpha *= 0.5
    return U, r

def main():
    # Main script with pseudo-arclength continuation
    x0 = 0.0
    L = 20.0 * np.pi
    x = np.linspace(x0, x1, 200)
    h = x[1] - x[0]
    N = len(x)
    b2 = 1.0
    r0 = -0.2  # initial bifurcation parameter
    beta = 0.0 # beta is the scale of the initial u
    u_0 = beta * np.sin(2 * np.pi * x / L)
    v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u
    U0 = np.concatenate([u0, v0])
    newton_tol = 1e-6
    max_newton_iter = 100

    ds = 0.1        # initial arclength step
    max_steps = 20  # maximum continuation steps

    solutions = np.zeros((len(U0) + 1, max_steps))
    tangents = np.zeros((len(U0) + 1, max_steps))

    F = lambda U, r: F_res(U, h, N, b2, r)
    U, residuals = newton_krylov(F, U0, newton_tol, max_newton_iter, h, N, b2, r0)
    solutions[:, 0] = np.concatenate([U, [r0]])

    initial_tangent = np.zeros(len(U0) + 1)
    initial_tangent[-1] = 1.0
    initial_tangent = initial_tangent / np.linalg.norm(initial_tangent)
    tangent = compute_tangent(U, r0, h, N, b2, initial_tangent)
    tangents[:, 0] = tangent

    for step in range(1, max_steps):
        U_prev = solutions[:-1, step-1]
        r_prev = solutions[-1, step-1]
        tangent_prev = tangents[:, step-1]
        U_pred = U_prev + ds * tangent_prev[:-1]
        r_pred = r_prev + ds * tangent_prev[-1]
        U_new, r_new = corrector(U_pred, r_pred, U_prev, r_prev, tangent_prev, ds, F, newton_tol, max_newton_iter, h, N, b2)
        solutions[:, step] = np.concatenate([U_new, [r_new]])
        tangent_new = compute_tangent(U_new, r_new, h, N, b2, tangent_prev)
        tangents[:, step] = tangent_new

    # Plot the bifurcation diagram: L2 norm of u vs r.
    # Extract only the steps that were used (nonzero columns)
    used = np.where(np.any(solutions, axis=0))[0]
    sol_data = solutions[:, used]
    r_values = sol_data[-1, :]
    u_values = sol_data[:N, :]
    l2_norms = np.linalg.norm(u_values, axis=0)

    plt.figure()
    plt.plot(r_values, l2_norms, marker='x')
    plt.xlabel('r')
    plt.ylabel(r'$\|u\|_{2}$')
    plt.title('Bifurcation Diagram: L2 Norm of u vs r')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()