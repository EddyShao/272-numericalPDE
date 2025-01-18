import numpy as np
import matplotlib.pyplot as plt

def GS(A, b, x0, tol, max_iter, bandwidth=None):
    n = len(b)
    x = x0
    hist = np.zeros((max_iter, n))
    if bandwidth is None:
        bandwidth = n
    for k in range(max_iter):
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, max(0, i-bandwidth):i], x[max(0, i-bandwidth):i]) - np.dot(A[i, i+1:min(i+1+bandwidth, n)], x[i+1:min(i+1+bandwidth, n)])) / A[i, i]
        hist[k, :] = x
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x, hist[:k+1, :]

    return x, hist


def Jacobi(A, b, x0, tol, max_iter, bandwidth=None):
    n = len(b)
    x = x0
    hist = np.zeros((max_iter, n))
    if bandwidth is None:
        bandwidth = n
    for k in range(max_iter):
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, max(0, i-bandwidth):i], x0[max(0, i-bandwidth):i]) - np.dot(A[i, i+1:min(i+1+bandwidth, n)], x0[i+1:min(i+1+bandwidth, n)])) / A[i, i]
        hist[k, :] = x
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x, hist[:k+1, :]
        x0 = x.copy()

    return x, hist


def SOR(A, b, x0, tol, max_iter, omega):
    n = len(b)
    x = x0
    hist = np.zeros((max_iter, n))
    for k in range(max_iter):
        for i in range(n):
            x[i] = (1 - omega) * x0[i] + omega * (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        hist[k, :] = x
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            return x, hist[:k+1, :]
        x0 = x

    return x, hist

if __name__ == '__main__':
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]])
    b = np.array([0, 5, 5, 0])
    x0 = np.zeros(4)
    tol = 1e-6
    max_iter = 1000
    x, hist = GS(A, b, x0, tol, max_iter)
    residual_hist = np.zeros(len(hist))
    for i in range(len(residual_hist)):
        residual_hist[i] = np.linalg.norm(np.dot(A, hist[i, :]) - b)
    print(x)
    print(np.linalg.solve(A, b))
    print(np.linalg.norm(x - np.linalg.solve(A, b)))
    print(np.dot(A, x) - b)
    plt.plot(residual_hist, label='GS')
    x0 = np.zeros(4)
    x, hist = Jacobi(A, b, x0, tol, max_iter)
    residual_hist = np.zeros(len(hist))
    for i in range(len(residual_hist)):
        residual_hist[i] = np.linalg.norm(np.dot(A, hist[i, :]) - b)
    print(x)
    print(np.linalg.solve(A, b))
    print(np.linalg.norm(x - np.linalg.solve(A, b)))
    print(np.dot(A, x) - b)

    print(residual_hist.shape)
    # make plot for jacobi iteration, and make sure it does not overlap with the previous plot
    plt.plot(residual_hist, label='Jacobi', color='red', linestyle='dashed', linewidth=2)
    plt.ylabel('Residual')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.show()