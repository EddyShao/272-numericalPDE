import numpy as np
from scipy.sparse import spdiags, lil_matrix, diags
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from utils import *

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


J = jacobian(U_0, h, N, b2, r=1.0)

J11 = J[:N, :N]
J12 = J[:N, N:]
J21 = J[N:, :N]
J22 = J[N:, N:]

J22_inv_J21 = spla.spsolve(J22, J21)
J_aux = J11 - J12 @ J22_inv_J21

eigs_1, _ = spla.eigs(J_aux, k=1, which='SR')

print(eigs_1)