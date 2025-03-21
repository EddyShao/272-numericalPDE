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
u_0 = 0.01 * np.sin(2 * np.pi * x / L)
v_0 = np.gradient(u_0, h) + u_0  # use h for gradient spacing


U_0 = np.concatenate((u_0, v_0))


results_seigs = []
results_leigs = []

for r in np.linspace(-4, 6, 41):
    # compute eigenvalues
    Jacobian = jacobian(U_0, h, N, b2, r)


    J11 = Jacobian[:N, :N]
    J12 = Jacobian[:N, N:]
    J21 = Jacobian[N:, :N]
    J22 = Jacobian[N:, N:]


    # implement the J11 +  J12 J22^-1 J21

    J22_inv_J21 = spla.spsolve(J22, J21)
    J_aux = J11 - J12 @ J22_inv_J21


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
plt.axvline(x=1, color='r', linestyle='--', label='r=1')
plt.xlabel('r')
plt.ylabel('eigenvalues')
plt.legend()
plt.title('Eigenvalues of Jacobian with respect to r')
plt.grid()
plt.savefig('p1.png')
plt.show(block=False)
plt.pause(2)
plt.close()








