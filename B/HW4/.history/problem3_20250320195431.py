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

u_0 = 0.01 * np.sin(2 * np.pi * x / L)
v_0 = np.gradient(u_0, h) + u_0  # use h for gradient spacing
U_0 = np.concatenate((u_0, v_0))


