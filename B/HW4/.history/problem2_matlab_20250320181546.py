import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *


def compute_tangent(U, r, h, N, b2, tangent_prev):
    """Compute the tangent vector for pseudo-arclength continuation."""
    J = jacobian(U, h, N, b2, r)
    Fr = np.zeros(2*N)
    u = U[:N]
    # Build M as in FEM
    M = assemble_mass(N, h)
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
    M = assemble_mass(N, h)

    for k in range(max_newton_iter):
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
        A_bottom = np.hstack([tangent_prev[:-1].reshape(1, -1), np.array([[tangent_prev[-1]]])])
        J_aug = np.vstack([A_top, A_bottom])
        delta = spla.gmres(sp.csr_matrix(J_aug), -residual, tol=1e-6, restart=200)[0]
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


# Main script with pseudo-arclength continuation
x0 = 0.0
L = 20 * np.pi
x = np.linspace(x0, L, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1.0
r0 = -0.2  # initial bifurcation parameter
beta = 0.0 # beta is the scale of the initial u
u0 = 0.000001 * np.sin(2 * np.pi * x / L)
v0 = np.gradient(u0, h) + u0  # use h for gradient spacing
U0 = np.concatenate([u0, v0])
newton_tol = 1e-6
max_newton_iter = 200

ds = 0.1        # initial arclength step
max_steps = 40  # maximum continuation steps

solutions = []
tangents = []

F = lambda U, r: F_res(U, h, N, b2, r)
U, residuals = newton_krylov(F, U0, newton_tol, max_newton_iter, h, N, b2, r0) # Initial solution
solutions.append(np.concatenate([U, [r0]]))


# we define initial tangent to be [0, \dots, -1]
tan_init = np.zeros(N+1)
tan_init[-1] = 1.0
tan_init /= np.linalg.norm(tan_init)
tangent = compute_tangent(U, r0, h, N, b2, tan_init)
tangents.append(tangent) 

for step in range(1, max_steps):
    sol = solutions[-1]
    U_prev = sol[:-1]
    r_prev = sol[-1]
    tan_last = tangents[-1]
    U_pred = U_prev + ds * tan_last[:-1]
    r_pred = r_prev + ds * tan_last[-1]
    U_new, r_new = corrector(U_pred, r_pred, U_prev, r_prev, tan_last, ds, F, newton_tol, max_newton_iter, h, N, b2)
    solutions.append(np.append(U_new, r_new))
    tan_new = compute_tangent(U_new, r_new, h, N, b2, tan_last)
    tangents.append(tan_new)


l2_norms = [np.linalg.norm(sol[:N]) for sol in solutions]
r_values = [sol[-1] for sol in solutions]

plt.figure()
plt.plot(r_values, l2_norms, marker='x')
plt.xlabel('r')
plt.ylabel(r'$\|u\|_{2}$')
plt.title('Bifurcation Diagram: L2 Norm of u vs r')
plt.grid(True)
plt.savefig('bifurcation_diagram.png')
plt.show(block=False)
plt.pause(2)
plt.close()


### Bisection search for the bifurcation point


def sign_Jacobian(U, r):
    Jacobian = jacobian(U, h, N, b2, r)
    J11 = Jacobian[:N, :N]
    J12 = Jacobian[:N, N:]
    J21 = Jacobian[N:, :N]
    J22 = Jacobian[N:, N:]
    J22_inv_J21 = spla.spsolve(J22, J21)
    J_aux = J11 - J12 @ J22_inv_J21
    eigs_1, _ = spla.eigs(J_aux, k=1, which='SR')
    eigs_2, _ = spla.eigs(-J_aux, k=1, which='SR')
    eigs_1, eigs_2 = np.real(eigs_1), np.real(eigs_2)
    # choose the one with smallest absolute value
    seigs = eigs_1 if np.abs(eigs_1) < np.abs(eigs_2) else eigs_2

    if seigs < 0:
        return -1
    else:
        return 1
    
# Bisection search

left = 0
right = len(solutions) - 1

while right - left > 1:
    mid = (left + right) // 2
    sign_mid = sign_Jacobian(solutions[mid][:N], solutions[mid][-1])
    sign_right = sign_Jacobian(solutions[right][:N], solutions[right][-1])
    if sign_mid * sign_right < 0:
        left = mid
    else:
        right = mid

