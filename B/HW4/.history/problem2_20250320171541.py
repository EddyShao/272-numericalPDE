import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from utils import *
import matplotlib.pyplot as plt

def compute_tangent_vector(U, h, N, b2, r):
    """Computes (u_dot, r_dot) by solving the augmented system."""
    J_U = jacobian(U, h, N, b2, r)  # Compute the Jacobian
    F_r = np.zeros(2 * N)  # Parameter derivative (approximate)
    M = assemble_mass(N, h)
    F_r[N:] = (M @ U[:N]).flatten()  # r derivative

    # Normalization row
    x_star = np.zeros(2 * N)
    x_star[0] = 1  # Fix first component

    # Construct augmented matrix
    A = np.vstack([np.hstack([J_U.toarray(), F_r[:, None]]), np.concatenate([x_star, np.array([1])])])

    # Define RHS
    b = np.zeros(2 * N + 1)
    b[-1] = 1  # Enforcing normalization condition

    # Solve for (u_dot, r_dot)
    sol = np.linalg.solve(A, b)
    
    return sol[:2*N], sol[-1]  # u_dot, r_dot

def arclength_continuation(U_0, h, N, b2, r, ds, num_steps, tol=1e-6, max_iter=100):
    """Performs pseudo-arclength continuation."""
    U = U_0.copy()
    solutions = [U.copy()]
    params = [r]

    for step in range(num_steps):
        print(f"Continuation step {step+1}")

        # Compute tangent vector (predictor direction)
        u_dot, r_dot = compute_tangent_vector(U, h, N, b2, r)
        tau = np.hstack([u_dot, r_dot])
        tau /= np.linalg.norm(tau)  # Normalize tangent

        # Predictor Step
        U_pred = U + ds * u_dot
        r_pred = r + ds * r_dot

        def augmented_residual(Ur):
            """Computes (F(U), arclength condition) for corrector step."""
            U_new = Ur[:2*N]
            r_new = Ur[-1]
            F_res = F_res(U_new, h, N, b2, r_new)

            # Arclength constraint: ||U - U_pred|| + ||r - r_pred|| - ds
            arc_res = np.linalg.norm(U_new - U_pred) + np.abs(r_new - r_pred) - ds
            return np.hstack([F_res, arc_res])

        def JVP(v):
            """Computes JVP using the augmented system."""
            J_U = jacobian(U, h, N, b2, r)
            J_v = J_U @ v[:2*N]
            arc_v = (np.dot(v[:2*N], U - U_pred) + v[-1] * (r - r_pred)) / np.linalg.norm(U - U_pred)
            return np.hstack([J_v, arc_v])

        # Define LinearOperator for GMRES
        J_op = LinearOperator((2*N+1, 2*N+1), matvec=JVP)

        # Corrector Step using GMRES
        U_corr = np.hstack([U_pred, r_pred])  # Initial guess
        for i in range(max_iter):
            res = augmented_residual(U_corr)
            norm_res = np.linalg.norm(res)
            print(f"Corrector Iteration {i}, ||Residual|| = {norm_res}")

            if norm_res < tol:
                break  # Converged

            dU, _ = gmres(J_op, -res, tol=tol)
            U_corr += dU  # Update solution

        # Update solution and parameter
        U = U_corr[:2*N]
        r = U_corr[-1]
        solutions.append(U.copy())
        params.append(r)

    return solutions, params

# Run arclength continuation
L = 20*np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 0. # SH23 parameter
r = -0.2 # bifurcation parameter

beta = 0.1 # beta is the scale of the initial u
u_0 = beta * np.sin(2 * np.pi * x / L)
v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u
U_0 = np.concatenate((u_0, v_0))

solutions, params = arclength_continuation(U_0, h, N, b2, r, ds=0.2, num_steps=10)


plt.plot(params, [np.linalg.norm(u[:N]) for u in solutions], marker='o')
plt.xlabel("Parameter r")
plt.ylabel("Solution Norm")
plt.title("Bifurcation Diagram")
plt.show()