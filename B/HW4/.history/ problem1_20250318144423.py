import numpy as np
import matplotlib.pyplot as plt



L =  1.
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 100)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter
beta = 0.1
u_0 = beta * np.sin(np.pi * x / L)
v_0 = beta * (np.pi/L)**2 * np.sin(np.pi * x / L) # v_0 =  \partial^{2} u / \partial x^{2} + u