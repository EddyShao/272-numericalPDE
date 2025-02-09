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


if __name__ == "__main__":
    L = 20 * np.pi  # Domain length
    N = 300  # Number of grid points
    x_space = np.linspace(0, L, N)  # Grid points
    u_1 = np.zeros(N)  # Initial guess
    u_2 = np.sin(2 * np.pi * x_space / L)  # Initial guess
    r = 0.0
    F = partial(F_u, r=r, h=h)
    r_1 = F(u_1)
    r_2 = F(u_2)
    # analytical solution
    r_1_a = np.zeros(N)
    F_2_a = lambda x: - (99 / 100)**2 * np.sin(2 * np.pi * x / L) -  np.sin(2 * np.pi * x / L) ** 3
    r_2_a = F_2_a(x_space)

    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    ax[0].plot(x_space, u_1, label=r'u_1')
    ax[0].plot(x_space, u_2, label=r'u_2')
    ax[0].set_title(r'u')
    ax[0].legend()

    ax[1].plot(x_space, r_1, label=r'F_1')
    ax[1].plot(x_space, r_2, label=r'F_2')
    ax[1].set_title(r'F - FDM')
    ax[1].legend()

    ax[2].plot(x_space, r_1_a, label=r'F_1')
    ax[2].plot(x_space, r_2_a, label=r'F_2')
    ax[2].set_title(r'F - analytical')
    ax[2].legend()

    plt.suptitle(r'Residual of Swift-Hohenberg equation')
    plt.savefig('residual.png')
    plt.show()