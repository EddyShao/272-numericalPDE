import numpy as np
import matplotlib.pyplot as plt



L =  1.
x_0 = 0
x_1 = L
x = np.linspace(x_0, x_1, 100)
h = x[1] - x[0]
N = len(x)
b2 = 1. # SH23 parameter