import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def F_u(u, r, h):
    """
    Compute the residual F(u) for the Swift-Hohenberg equation in 1D using FDM.
    
    Parameters:
        u (numpy.ndarray): State vector of size N.
        r (float): Bifurcation parameter.
        L (float): Domain length.
        N (int): Number of grid points.
    
    Returns:
        numpy.ndarray: Residual vector of the same size as u.
    """
    F = np.zeros_like(u)
    
    # Compute second derivative using central finite difference
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / h**2
    
    # Compute fourth derivative using central finite difference
    u_xxxx = np.zeros_like(u)
    u_xxxx[2:-2] = (u[:-4] - 4*u[1:-3] + 6*u[2:-2] - 4*u[3:-1] + u[4:]) / h**4
    
    # Compute the residual
    F = r * u - (u + 2 * u_xx + u_xxxx) - u**3
    
    # Apply natural boundary conditions (Neumann BCs: u_xx = u_xxxx = 0 at boundaries)
    F[0] = 0  # Approximate boundary values
    F[-1] = 0
    
    return F