import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


### We build the tangent vector based on the following, which is listed in the pset ###


# The tangent vector $\dot{x}$ at a solution point $(u, r)$ satisfies:
# \[
# \begin{pmatrix}
# F_u & F_r \\
# \dot{x}_*^T & 0
# \end{pmatrix}
# \begin{pmatrix}
# \dot{u} \\
# \dot{r}
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0 \\
# 1
# \end{pmatrix}
# \]
# where $F_u$ is the Jacobian matrix and $F_r$ is the parameter derivative.


def compute_tangent(U, r, h, N, b2, tangent_prev):
    F_U = J(U, h, N, b2, r)
    F_r = np.zeros(2*N)
    M = assemble_mass(N, h)
    F_r[N:] = M @ U[:N]

    x_dot = np.zeros(2*N+1)
    x_dot[0] = 1.
    J_aug = np.vstack([sp.vstack([J, Fr]), np.hstack([x_dot, 0])])
    b = np.zeros(2*N+1)
    b[-1] = 1.
    tangent = np.linalg.solve(J_aug, b)
    
    return tangent

# Corrector step
def corrector(U_pred, r_pred, U_prev, r_prev, tangent_prev, ds, F, tol, max_iter, h, N, b2):
    U, r = U_pred.copy(), r_pred
    n = len(U) // 2

    for _ in range(max_iter):
        Fx = F(U, r)
        arclength_res = np.dot(tangent_prev, np.hstack([U - U_prev, r - r_prev])) - ds
        residual = np.hstack([Fx, arclength_res])

        if np.linalg.norm(residual) < tol:
            break

        J = jacobian(U, h, N, b2, r)
        Fr = np.zeros(len(U))
        Fr[:n] = -(h/6) * (4*U[:n] + 2*U[:n])

        J_aug = np.vstack([sp.vstack([J, Fr]), np.hstack([tangent_prev[:-1], tangent_prev[-1]])])
        delta = np.linalg.solve(J_aug, -residual)

        delta_U, delta_r = delta[:-1], delta[-1]

        alpha = 1.0
        for _ in range(10):
            U_temp, r_temp = U + alpha * delta_U, r + alpha * delta_r
            if np.linalg.norm(np.hstack([F(U_temp, r_temp), np.dot(tangent_prev, np.hstack([U_temp - U_prev, r_temp - r_prev])) - ds])) < np.linalg.norm(residual):
                U, r = U_temp, r_temp
                break
            alpha *= 0.5

    return U, r

# Main pseudo-arclength continuation script
L = 20*np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter
r = -0.0 # bifurcation parameter

#### Newton-Krylov parameters ####
TOL = 1e-10
MAX_ITER = 1000

#### Initial condition ####
beta = 20 # beta is the scale of the initial u
u_0 = beta * np.sin(2 * np.pi * x / L)
v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u


U_0 = np.concatenate((u_0, v_0))

# Solve using Newton-Krylov with self-implemented newton_krylov, gmres is imported from scipy.sparse.linalg
U_sol = newton_krylov(U_0, h, N, b2, r, TOL, MAX_ITER)
u = U_sol[:N]


tangent = np.zeros(len(U0) + 1)
tangent[-1] = 1
tangent /= np.linalg.norm(tangent)
tangents[:, 0] = tangent

for step in range(1, max_steps):
    U_prev, r_prev = solutions[:-1, step-1], solutions[-1, step-1]
    tangent_prev = tangents[:, step-1]
    U_pred, r_pred = U_prev + ds * tangent_prev[:-1], r_prev + ds * tangent_prev[-1]

    U_new, r_new = corrector(U_pred, r_pred, U_prev, r_prev, tangent_prev, ds, F, newton_tol, max_newton_iter, h, N, b2)
    solutions[:, step] = np.hstack([U_new, r_new])

    tangents[:, step] = compute_tangent(U_new, r_new, h, N, b2, tangents[:, step-1])

# Plot bifurcation diagram
import matplotlib.pyplot as plt
r_values = solutions[-1, :]
l2_norms = np.linalg.norm(solutions[:N, :], axis=0)
plt.plot(r_values, l2_norms, 'b-', linewidth=2)
plt.xlabel('r (Bifurcation Parameter)')
plt.ylabel('L2 Norm of u')
plt.title('Bifurcation Diagram')
plt.grid()
plt.show()