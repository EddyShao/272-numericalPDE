import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *


##### We first put the pseudo-code here

# # ---- Step 1: Compute critical eigenvector φ and adjoint eigenvector ψ ----
# Compute the Jacobian matrix at $r=0$, and trivial solution $u=0$.
# (J11, J12, J21, J22) = jacobian(u_current, r_current)
# Define Schur's complememnt as J_prime = J11 - J12 @ J22^-1 @ J21
# compute the eigenvector corresponding to zero for both J_prime and J_prime^T
# denoted them as psi and phi
# Normalize them

# # ---- Step 3: Compute normal form coefficients ----
# # Typically involves second & third derivative terms D^2F, D^3F, projected onto φ, ψ
# (alpha, beta) = compute_normal_form_coefficients(u_current, r_current, φ, ψ)
# # The sign of beta often determines ± x^3 in the normal form

# # ---- Step 4: Branch switching initial guess ----
# sign_cubic = sign(beta)  # ± 1
# ε = small_amplitude
# Δr = small_parameter_step * sign_cubic
# u_switch_init = u_current + ε * φ
# r_switch_init = r_current + Δr

# # ---- Step 5: Continuation on the new branch ----
# (u_solution_array, r_values) = []
# (u0, r0) = (u_switch_init, r_switch_init)
# for step in 1..max_steps:
# # predictor:
# (u_pred, r_pred) = predictor(u0, r0, ...)

# # corrector with Newton:
# (u_new, r_new) = newton_corrector(u_pred, r_pred, ...)

# # store solution
# u_solution_array.append(u_new)
# r_values.append(r_new)

# # optionally compute new tangent (pseudo-arclength) if using arclength method
# # (u0, r0) = (u_new, r_new)

# # ---- Step 6: Plot snaking diagram ----
# # e.g. L2 norm vs. r
# norms = [ L2_norm(u_sol) for u_sol in u_solution_array ]
# plot(r_values, norms)
# show()

#### SH23 parameters ####
L = 20 * np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter

u_0 = np.zeros(N)
v_0 = np.gradient(u_0, h) + u_0  # use h for gradient spacing
U_0 = np.concatenate((u_0, v_0))


### 

J = jacobian(U_0, h, N, b2, r=1.0)

J11 = J[:N, :N]
J12 = J[:N, N:]
J21 = J[N:, :N]
J22 = J[N:, N:]

J22_inv_J21 = spla.spsolve(J22, J21)
J_aux = J11 - J12 @ J22_inv_J21

eigs_1, phi = spla.eigs(J_aux, k=1, which='SR')
eigs_2, psi = spla.eigs(J_aux.T, k=1, which='SR')

phi = phi.real.flatten()
psi = psi.real.flatten()

# normalize phi and psi
phi /= np.linalg.norm(phi)
psi /= np.linalg.norm(psi)



# projection onto the system
# using the normal form
# it shold return:
#       alpha = \langle \psi, D^2F(u_star)[phi,phi] \rangle
#       beta  = \langle \psi, D^3F(u_star)[phi,phi,phi] \rangle
#       sigma = \langle \psi, dF/dr(u_star) \rangle


def v(u):
    # v = Lap u  + u
    return np.gradient(u, h) + u




# first order derivative
U_center = np.concatenate((u_0, v(u_0)))
res_center = F_res(U_0, h, N, b2, r=1.0)

# let eps be machine epsilon
eps = 40 * np.finfo(float).eps
print('machine epsilon', eps)
U_right = np.concatenate((u_0 + eps * phi, v(u_0 + eps * phi)))
res_right = F_res(U_right, h, N, b2, r=1.0)

U_left = np.concatenate((u_0 - eps * phi, v(u_0 - eps * phi)))
res_left = F_res(U_left, h, N, b2, r=1.0)

first_derivative = (res_right - res_left)[:N] / eps

alpha = np.dot(psi, first_derivative) 
print(alpha)




