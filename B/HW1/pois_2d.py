import numpy as np
import matplotlib.pyplot as plt
from iterativeMethods import GS, Jacobi, SOR

# implement solver of 2D Poisson equation with GS, Jacobi, and SOR

def poisson_2d(f, n, tol, max_iter, method, omega=None):
    h = 1/(n+1)
    x = np.linspace(0, 1, n+2)
    y = np.linspace(0, 1, n+2)
    X, Y = np.meshgrid(x[1:-1], y[1:-1])
    stencil = np.vstack((X.ravel(), Y.ravel())).T

    # f = lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    F = f(stencil.T)
    F = F

    A = np.eye(n**2)*(-4)\
         + np.eye(n**2, k=1)\
         + np.eye(n**2, k=-1)\
         + np.eye(n**2, k=n)\
         + np.eye(n**2, k=-n)
    A = A/h**2
    x_init = np.ones(n**2)

    if method == 'GS':
        u, hist = GS(A, F, x_init, tol, max_iter, bandwidth=n)
    elif method == 'Jacobi':
        u, hist = Jacobi(A, F, x_init, tol, max_iter, bandwidth=n)
    elif method == 'SOR':
        u, hist = SOR(A, F, x_init, omega, tol, max_iter, bandwidth=n)
    elif method == 'direct':
        u = np.linalg.solve(A, F)
        hist = None
    else:
        raise ValueError('method should be GS, Jacobi, or SOR')
    
    # reshape the solution to 2D
    # according to the correct order of the meshgrid
    u = u.reshape((n, n))
    # add boundary conditions by padding zeros on the boundary
    u = np.pad(u, ((1, 1), (1, 1)), 'constant', constant_values=0.)
    residual_hist = np.zeros(len(hist))
    for i in range(len(residual_hist)):
        residual_hist[i] = np.linalg.norm(np.dot(A, hist[i, :]) - F)
    return u, residual_hist

if __name__ == '__main__':
    
    tol = 1e-6
    max_iter = 50000
    omega = 1.5 # it is for SOR; not used for GS and Jacobi
    f1 = lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    f2 = lambda x: np.where(np.max(np.abs(x - 0.5), axis=0) <= 0.1, 1.0, 0.0)

    hist_1_GS = {}
    hist_1_J = {}
    hist_2_GS = {}
    hist_2_J = {}

    n_list = [16, 32, 64]

    for n in n_list:
        u, hist_1_GS[n] = poisson_2d(f1, n, tol, max_iter, 'GS', omega)
        u, hist_1_J[n] = poisson_2d(f1, n, tol, max_iter, 'Jacobi', omega)

        u, hist_2_GS[n] = poisson_2d(f2, n, tol, max_iter, 'GS', omega)
        u, hist_2_J[n] = poisson_2d(f2, n, tol, max_iter, 'Jacobi', omega)
        print('n =', n, 'done')

    # plot the convergence history for GS and Jacobi
    # for the two different right-hand sides
    # plot 2 seperate subplots for each right-hand side
    # Also for n = 16, 32, 64 seperately
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, n in enumerate(n_list):
        axs[0, i].plot(hist_1_GS[n], label='GS')
        axs[0, i].plot(hist_1_J[n], label='Jacobi')
        axs[0, i].set_title('f1, n = {}'.format(n))
        axs[0, i].set_xlabel('Iteration')
        axs[0, i].set_ylabel(r'Residual $\|Ax - b\|_{2}$')
        axs[0, i].set_yscale('log')
        axs[0, i].legend()

        axs[1, i].plot(hist_2_GS[n], label='GS')
        axs[1, i].plot(hist_2_J[n], label='Jacobi')
        axs[1, i].set_title('f2, n = {}'.format(n))
        axs[1, i].set_xlabel('Iteration')
        axs[1, i].set_ylabel(r'Residual $\|Ax - b\|_{2}$')
        axs[1, i].set_yscale('log')
        axs[1, i].legend()

    plt.tight_layout()
    plt.savefig('poisson_2d_convergence.png')
    plt.show()
    np.savez('poisson_2d_convergence.npz', hist_1_GS=hist_1_GS, hist_1_J=hist_1_J, hist_2_GS=hist_2_GS, hist_2_J=hist_2_J)
