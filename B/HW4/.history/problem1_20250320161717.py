import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *

# Solve the system using Newton-Krylov



#### SH23 parameters ####
L = 20 * np.pi
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 200)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter


#### Newton-Krylov parameters ####
TOL = 1e-10
MAX_ITER = 1000

#### Initial condition ####
beta = 0.0 # beta is the scale of the initial u
u_0 = beta * np.sin(2 * np.pi * x / L)
v_0 = - (2 * np.pi/L)**2 * u_0 + u_0  # v_0 =  \partial^{2} u / \partial x^{2} + u


U_0 = np.concatenate((u_0, v_0))


results_seigs = []
results_leigs = []

for r in np.linspace(-4, 6, 41):
    # compute eigenvalues
    Jacobian = J(U_0, h, N, b2, r)


    J11 = Jacobian[:N, :N]
    J12 = Jacobian[:N, N:]
    J21 = Jacobian[N:, :N]
    J22 = Jacobian[N:, N:]


    # implement the J11 +  J12 J22^-1 J21

    J22_inv_J21 = spla.spsolve(J22, J21)
    J_aux = J11 + J12 @ J22_inv_J21


    eigs_1, _ = spla.eigs(J_aux, k=1, which='SR')
    eigs_2, _ = spla.eigs(-J_aux, k=1, which='SR')
    eigs_1, eigs_2 = np.real(eigs_1), np.real(eigs_2)
    # choose the one with smallest absolute value
    seigs = eigs_1 if np.abs(eigs_1) < np.abs(eigs_2) else eigs_2
    results_seigs.append(seigs)
    leigs = spla.eigs(J_aux, k=1, which='LR')[0].real
    results_leigs.append(leigs)
    

plt.plot(np.linspace(-4, 6, 41), results_seigs, marker='x', label='eigenvalues with smallest abs real part')
plt.plot(np.linspace(-4, 6, 41), results_leigs, marker='o', label='eigenvalues with largest real part')
# draw line y = 0
plt.axhline(y=0, color='r', linestyle='--', label='eig=0')
plt.xlabel('r')
plt.ylabel('eigenvalues')
plt.legend()
plt.title('Eigenvalues of Jacobian with respect to r')
plt.savefig('p1.png')
plt.show()
plt.pause(0.1)








