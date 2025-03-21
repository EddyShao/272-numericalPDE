import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# RE-USE YOUR EXISTING PDE / ASSEMBLY / JACOBIAN FUNCTIONS
# ---------------------------------------------------------------------
def F_res(U, h, N, b2, r):
    """
    PDE residual function F(U,r).
    U is a 2*N array (u and v stacked),
    b2 and r are parameters.
    """
    # -- Example from your code (adapt to your PDE) --
    e = np.ones(N)
    A = sp.diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N)) * (1/h)
    A = A.tolil()
    # Periodic BC adjustments
    A[N-1, 0] = -1/h
    A[0, N-1] = -1/h
    A = A.tocsr()
    
    M = assemble_mass(N, h)
    
    u = U[:N]
    v = U[N:]
    f1 = (M + A).dot(u) - M.dot(v)
    # For example, second eqn is: v' + M * (r*u + b2*u^2 - u^3)
    f2 = -M.dot(r*u + b2*u**2 - u**3) + (M + A).dot(v)
    return np.concatenate([f1, f2])

def jacobian(U, h, N, b2, r):
    """
    Sparse Jacobian of F w.r.t. U.
    """
    e = np.ones(N)
    A = sp.diags([-e, 2*e, -e], [-1, 0, 1], shape=(N, N)) * (1/h)
    A = A.tolil()
    A[N-1, 0] = -1/h
    A[0, N-1] = -1/h
    A = A.tocsr()
    
    M = assemble_mass(N, h)
    
    u = U[:N]
    v = U[N:]
    # Blocks
    J11 = (M + A)
    J12 = -M
    diag_term = (r + 2*b2*u - 3*u**2)
    J21 = -M.dot(sp.diags(diag_term, 0))
    J22 = (M + A)
    
    top = sp.hstack([J11, J12])
    bottom = sp.hstack([J21, J22])
    J = sp.vstack([top, bottom])
    return J.tocsr()

def assemble_mass(N, h):
    """
    Mass matrix M. 
    """
    e = np.ones(N)
    M = sp.diags([e, 4*e, e], [-1,0,1], shape=(N,N))*(h/6)
    M = M.tolil()
    # Periodic
    M[N-1, 0] = h/6
    M[0, N-1] = h/6
    return M.tocsr()

# ---------------------------------------------------------------------
# EXISTING NEWTON SOLVER / KRYLOV
# ---------------------------------------------------------------------
def newton_krylov(F, U0, newton_tol, max_newton_iter, h, N, b2, r):
    U = U0.copy()
    residuals = []
    for k in range(max_newton_iter):
        Fx = F(U, r)
        normFx = np.linalg.norm(Fx)
        residuals.append(normFx)
        if normFx < newton_tol:
            break
        J = jacobian(U, h, N, b2, r)
        dU, info = spla.gmres(J, -Fx, tol=1e-6, restart=200)
        # simple line search
        alpha = 1.0
        for _ in range(10):
            U_test = U + alpha*dU
            Fx_test = F(U_test, r)
            if np.linalg.norm(Fx_test) < normFx:
                U = U_test
                break
            alpha *= 0.5
    return U, residuals

# ---------------------------------------------------------------------
# TANGENT + CORRECTOR (PSEUDO-ARCLENGTH)
# ---------------------------------------------------------------------
def compute_tangent(U, r, h, N, b2, tangent_prev):
    """Compute the tangent vector for pseudo-arclength continuation."""
    J = jacobian(U, h, N, b2, r)
    Fr = np.zeros(2*N)
    u = U[:N]
    M = assemble_mass(N, h)
    Fr[:N] = -M.dot(u)
    dr_prev = tangent_prev[-1]
    if abs(dr_prev) < 1e-10:
        dr_prev = 1.0
    dU_temp = spla.spsolve(J, -Fr*dr_prev)
    tangent_temp = np.concatenate([dU_temp, [dr_prev]])
    # Augmented system
    J_dense = J.toarray()
    A_top = np.hstack([J_dense, Fr.reshape(-1, 1)])
    A_bottom = np.hstack([tangent_temp[:-1].reshape(1, -1), np.array([[tangent_temp[-1]]])])
    A = np.vstack([A_top, A_bottom])
    b_vec = np.concatenate([np.zeros(2*N), [1]])
    tangent = np.linalg.solve(A, b_vec)
    if np.any(np.isnan(tangent)) or np.any(np.isinf(tangent)):
        raise ValueError("Tangent computation failed: NaN or Inf.")
    return tangent

def corrector(U_pred, r_pred, U_prev, r_prev, tangent_prev, ds, F, newton_tol, max_newton_iter, h, N, b2):
    """Newton corrector step with arclength constraint."""
    U = U_pred.copy()
    r = r_pred
    M = assemble_mass(N, h)
    for _ in range(max_newton_iter):
        Fx = F(U, r)
        arc_res = np.dot(tangent_prev, np.concatenate([U, [r]]) - np.concatenate([U_prev, [r_prev]])) - ds
        residual = np.concatenate([Fx, [arc_res]])
        if np.linalg.norm(residual) < newton_tol:
            break
        J = jacobian(U, h, N, b2, r)
        Fr = np.zeros_like(U)
        u = U[:N]
        Fr[:N] = -M.dot(u)
        J_dense = J.toarray()
        A_top = np.hstack([J_dense, Fr.reshape(-1, 1)])
        A_bottom = np.hstack([tangent_prev[:-1].reshape(1, -1),
                              np.array([[tangent_prev[-1]]])])
        J_aug = np.vstack([A_top, A_bottom])
        delta = spla.gmres(sp.csr_matrix(J_aug), -residual, tol=1e-6, restart=200)[0]
        delta_U = delta[:-1]
        delta_r = delta[-1]
        alpha = 1.0
        Fx_norm = np.linalg.norm(residual)
        for __ in range(10):
            U_test = U + alpha*delta_U
            r_test = r + alpha*delta_r
            Fx_test = F(U_test, r_test)
            arc_test = np.dot(tangent_prev,
                              np.concatenate([U_test, [r_test]]) -
                              np.concatenate([U_prev, [r_prev]])) - ds
            new_res = np.concatenate([Fx_test, [arc_test]])
            if np.linalg.norm(new_res) < Fx_norm:
                U = U_test
                r = r_test
                break
            alpha *= 0.5
    return U, r

# ---------------------------------------------------------------------
# BIFURCATION DETECTION UTILITIES
# ---------------------------------------------------------------------
def compute_largest_real_eigs(U, r, h, N, b2, k=3):
    """
    Compute k eigenvalues with the largest real part.
    Returns a sorted array of real parts, descending.
    """
    J = jacobian(U, h, N, b2, r)
    # Use scipy.sparse.linalg.eigs with 'LR' (largest real part)
    # For large problems, you may need to refine or use shift-invert.
    vals, _ = spla.eigs(J, k=k, which='LR')
    return np.sort(vals.real)[::-1]  # sort descending by real part

def sign_changes(eigvals_old, eigvals_new):
    """
    Check if any eigenvalue has changed sign between two sets of eigenvalues.
    Return True if sign change in real part is detected.
    """
    # Compare by matching sorted order:
    # We'll do a simple check: if the number of positive eigenvalues differs, there's a sign change.
    pos_old = np.sum(eigvals_old > 0)
    pos_new = np.sum(eigvals_new > 0)
    return (pos_old != pos_new)

def locate_bifurcation(U_left, r_left, U_right, r_right, F, h, N, b2,
                       max_iter=10, k=3, tol=1e-8):
    """
    Bisection-like approach to refine the parameter r where an eigenvalue crosses zero.
    We assume sign_changes(...) is True between r_left and r_right.
    """
    for _ in range(max_iter):
        r_mid = 0.5*(r_left + r_right)
        # Solve PDE at r_mid with Newton
        U_mid, _ = newton_krylov(F, 0.5*(U_left+U_right), tol, 30, h, N, b2, r_mid)
        # Compute eigenvalues
        eig_mid = compute_largest_real_eigs(U_mid, r_mid, h, N, b2, k=k)
        if sign_changes(compute_largest_real_eigs(U_left, r_left, h, N, b2, k=k),
                        eig_mid):
            # Bifurcation in [r_left, r_mid]
            U_right = U_mid
            r_right = r_mid
        else:
            # Bifurcation in [r_mid, r_right]
            U_left = U_mid
            r_left = r_mid
        if abs(r_right - r_left) < tol:
            break
    return 0.5*(r_left + r_right), 0.5*(U_left + U_right)

# ---------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------
def main():
    # Setup
    x0 = 0.0
    L = 20*np.pi
    num_points = 200
    x = np.linspace(x0, L, num_points)
    h = x[1] - x[0]
    N = len(x)
    b2 = 1.0
    r0 = -0.2
    newton_tol = 1e-6
    max_newton_iter = 200
    ds = 0.1
    max_steps = 40

    # Initial guess
    u0 = 1e-6*np.sin(2*np.pi*x/L)
    v0 = np.gradient(u0, h) + u0
    U0 = np.concatenate([u0, v0])

    # PDE residual
    F = lambda U, r: F_res(U, h, N, b2, r)

    # Solve PDE at (U0, r0)
    U_sol, _ = newton_krylov(F, U0, newton_tol, max_newton_iter, h, N, b2, r0)

    # Store solutions
    solutions = [np.append(U_sol, r0)]
    # Initial tangent
    tan_init = np.zeros_like(solutions[-1])
    tan_init[-1] = 1.0
    tan_init /= np.linalg.norm(tan_init)
    tangent = compute_tangent(U_sol, r0, h, N, b2, tan_init)
    tangents = [tangent]

    # Arrays to track eigenvalues
    largest_eigs_list = []
    # Compute initial eigenvalues
    init_eigs = compute_largest_real_eigs(U_sol, r0, h, N, b2, k=3)
    largest_eigs_list.append(init_eigs)

    # Continuation
    for step in range(1, max_steps):
        sol_old = solutions[-1]
        U_prev = sol_old[:-1]
        r_prev = sol_old[-1]
        tan_old = tangents[-1]

        # Predictor
        U_pred = U_prev + ds*tan_old[:-1]
        r_pred = r_prev + ds*tan_old[-1]

        # Corrector
        U_new, r_new = corrector(U_pred, r_pred, U_prev, r_prev, tan_old, ds,
                                 F, newton_tol, max_newton_iter, h, N, b2)
        solutions.append(np.append(U_new, r_new))
        # New tangent
        tan_new = compute_tangent(U_new, r_new, h, N, b2, tan_old)
        tangents.append(tan_new)

        # Compute largest-real-part eigenvalues & check sign changes
        eigs_new = compute_largest_real_eigs(U_new, r_new, h, N, b2, k=3)
        largest_eigs_list.append(eigs_new)

        if sign_changes(largest_eigs_list[-2], eigs_new):
            print(f"Potential bifurcation between step {step-1} and step {step}!")
            # Bisection to refine the crossing
            sol_left = solutions[-2]
            U_left, r_left = sol_left[:-1], sol_left[-1]
            sol_right = solutions[-1]
            U_right, r_right = sol_right[:-1], sol_right[-1]

            r_bif, U_bif = locate_bifurcation(U_left, r_left, U_right, r_right,
                                              F, h, N, b2, max_iter=10, k=3)
            print(f"  Bifurcation located near r = {r_bif:.6f}")
            # (Optional) you can store or refine solution further if desired

    # Plot the bifurcation diagram
    l2_norms = [np.linalg.norm(sol[:N]) for sol in solutions]
    r_values = [sol[-1] for sol in solutions]

    plt.figure(figsize=(6,5))
    plt.plot(r_values, l2_norms, marker='x', label='Solution branch')
    plt.xlabel('r')
    plt.ylabel(r'$\|u\|_{2}$')
    plt.title('Bifurcation Diagram (L2 norm vs. r)')
    plt.grid(True)
    plt.legend()
    plt.savefig('bifurcation_diagram.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Plot the tracked eigenvalues vs. r
    # largest_eigs_list is shape [step][k], each step we have 3 eigenvalues
    r_values = np.array(r_values)
    largest_eigs_list = np.array(largest_eigs_list)  # shape (steps, k=3)

    plt.figure(figsize=(6,5))
    for i in range(largest_eigs_list.shape[1]):
        plt.plot(r_values, largest_eigs_list[:, i],
                 marker='o', label=f'Eig {i+1}')
    plt.axhline(0, color='k', ls='--', label='0')
    plt.xlabel('r')
    plt.ylabel('Real part of eigenvalues')
    plt.title('Largest-Real-Part Eigenvalues vs. r')
    plt.legend()
    plt.grid(True)
    plt.savefig('eigs_vs_r.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

if __name__ == '__main__':
    main()