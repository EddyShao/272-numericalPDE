import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


##############################################################################
# Find Bifurcation: We assume there's a trivial solution (u=0).
def trivial_solution(N):
    # The trivial solution is u=0
    return np.zeros(N)

def detect_bifurcation(u_trivial, r_values, D, N, dx):
    """
    Suppose we do a sweep in r_values, and each time we check
    the smallest-magnitude eigenvalue of the Jacobian at (u=0).
    If an eigenvalue crosses zero, we found a candidate r*.
    Return (u_star, r_star).
    """
    # For u=0 => J(0,r)= r*I + D*L
    # The eigenvalues of L are known analytically for periodic BC,
    # but let's do it numerically for demonstration.
    for i in range(len(r_values)-1):
        rA = r_values[i]
        rB = r_values[i+1]
        # Check sign of smallest real eigenvalue at rA vs rB
        evalsA = compute_smallest_abs_eig(u_trivial, rA, D, N, dx)
        evalsB = compute_smallest_abs_eig(u_trivial, rB, D, N, dx)
        # If sign changes => potential crossing
        if evalsA*evalsB < 0:
            # We'll just pick the midpoint
            r_star = 0.5*(rA + rB)
            u_star = trivial_solution(N)
            return (u_star, r_star)
    return None, None

def compute_smallest_abs_eig(u, r, D, N, dx):
    """
    Compute the real eigenvalue of J(u,r) with smallest absolute value.
    (For demonstration only, in practice you'd do a more robust approach.)
    """
    J = assemble_jacobian(u, r, D, N, dx)
    # ARPACK to find a few eigenvalues near 0 (shift-invert sigma=0)
    try:
        vals, _ = spla.eigs(J, k=2, sigma=0.0, which='LM')
        # Return real part of the eigenvalue with smallest abs
        idx = np.argmin(np.abs(vals))
        return vals[idx].real
    except:
        return np.inf

##############################################################################
# 4) Compute Null Vector phi and Adjoint Vector psi
##############################################################################
def compute_null_and_adjoint(u_star, r_star, D, N, dx):
    """
    Solve J(u_star, r_star)*phi=0 and J(u_star, r_star)^T * psi=0
    in discrete PDE sense.
    """
    J = assemble_jacobian(u_star, r_star, D, N, dx)
    # 4.1: Null vector phi
    # Solve J*phi=0. Use an eigensolver with shift=0 or a direct approach.
    vals, vecs = spla.eigs(J, k=1, sigma=0.0, which='LM')
    phi = vecs[:, 0].real  # might need to pick the correct eigenvector
    # Normalize
    norm_phi = np.sqrt(np.sum(phi**2))
    phi /= norm_phi

    # 4.2: Adjoint => J^T*psi=0
    JT = J.transpose().tocsr()
    valsT, vecsT = spla.eigs(JT, k=1, sigma=0.0, which='LM')
    psi = vecsT[:, 0].real
    # Normalize so that <psi, phi> = 1 (inner product = sum(psi[i]*phi[i])
    dot_val = np.sum(psi*phi)
    psi /= dot_val
    return phi, psi

##############################################################################
# 5) Compute Normal Form Coefficients alpha, beta, sigma
##############################################################################
def compute_normal_form_coeffs(u_star, r_star, phi, psi, D, N, dx):
    """
    alpha = <psi, D^2F(u_star)[phi, phi]>
    beta  = <psi, D^3F(u_star)[phi, phi, phi]>
    sigma = <psi, dF/dr(u_star)>
    
    We'll approximate these by finite differences in the PDE context.
    For a real PDE, you'd do a more direct approach or symbolic expansions.
    """
    # 5.1: function that returns F(u,r)
    def F(u, r):
        return assemble_pde_residual(u, r, D, N, dx)
    
    # 5.2: approximate D^2F(u_star)[phi, phi]
    # We'll do a second-order finite difference:
    eps = 1e-6
    F0 = F(u_star, r_star)
    Fp = F(u_star + eps*phi, r_star)
    Fm = F(u_star - eps*phi, r_star)
    # 2nd derivative in direction phi => (Fp - 2F0 + Fm)/(eps^2)
    d2 = (Fp - 2*F0 + Fm)/(eps**2)
    alpha = np.dot(psi, d2)

    # 5.3: approximate D^3F(u_star)[phi, phi, phi]
    # We'll do one more difference on d2:
    # d2(u + eps*phi) - 2*d2(u) + d2(u - eps*phi)
    # but each d2(...) is the second derivative. We'll define a helper:
    def second_derivative(u):
        # re-implement the logic from above for the direction phi
        F0 = F(u, r_star)
        Fp = F(u + eps*phi, r_star)
        Fm = F(u - eps*phi, r_star)
        return (Fp - 2*F0 + Fm)/(eps**2)
    
    d2_0 = second_derivative(u_star)
    d2_p = second_derivative(u_star + eps*phi)
    d2_m = second_derivative(u_star - eps*phi)
    d3 = (d2_p - 2*d2_0 + d2_m)/(eps**2)
    beta = np.dot(psi, d3)

    # 5.4: partial wrt r => <psi, dF/dr(u_star)>
    # We'll do a finite difference in r:
    F_rp = F(u_star, r_star + eps)
    F_rm = F(u_star, r_star - eps)
    dF_dr_approx = (F_rp - F_rm)/(2*eps)
    sigma = np.dot(psi, dF_dr_approx)
    return alpha, beta, sigma

##############################################################################
# 6) Branch Switching
##############################################################################
def branch_switch(u_star, r_star, phi, beta, eps=1e-3):
    """
    For a pitchfork, the sign of beta decides the direction in r.
    We'll do:
      u_switch = u_star + eps*phi
      r_switch = r_star + sign(beta)*eps
    """
    sign_beta = np.sign(beta) if beta!=0 else 1.0
    u_switch = u_star + eps*phi
    r_switch = r_star + sign_beta*eps
    return u_switch, r_switch

##############################################################################
# 7) Demo "main" that ties it all together
##############################################################################
def main():
    # Setup domain
    L = 2*np.pi
    N = 64
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    D = 1.0
    
    # We suspect a pitchfork near r=0, with trivial solution u=0.
    # Let's do a parameter sweep in [r_min, r_max].
    r_values = np.linspace(-1.0, 1.0, 20)
    u_triv = trivial_solution(N)  # all zeros
    
    # Detect approximate bifurcation
    u_star, r_star = detect_bifurcation(u_triv, r_values, D, N, dx)
    if u_star is None:
        print("No bifurcation found in that range.")
        return
    print(f"Detected potential pitchfork at r*={r_star:.4f}")
    
    # Double-check by solving PDE at (u_star, r_star)
    u_star_sol = newton_solve(u_star, r_star, D, N, dx)
    print(f"||F(u_star_sol)||={np.linalg.norm(assemble_pde_residual(u_star_sol, r_star, D, N, dx)):.2e}")
    
    # Compute null vector phi and adjoint psi
    phi, psi = compute_null_and_adjoint(u_star_sol, r_star, D, N, dx)
    print("Computed null vector phi and adjoint psi.")
    
    # Compute normal form coefficients
    alpha, beta, sigma = compute_normal_form_coeffs(u_star_sol, r_star, phi, psi, D, N, dx)
    print(f"Normal Form Coeffs near (u*,r*): alpha={alpha:.4e}, beta={beta:.4e}, sigma={sigma:.4e}")
    
    # Branch switch
    u_switch, r_switch = branch_switch(u_star_sol, r_star, phi, beta, eps=1e-3)
    print(f"Switching to new branch with initial guess r={r_switch:.4f}")
    
    # Solve PDE at the switch guess
    u_new_branch = newton_solve(u_switch, r_switch, D, N, dx)
    print(f"||F(u_new_branch)||={np.linalg.norm(assemble_pde_residual(u_new_branch, r_switch, D, N, dx)):.2e}")
    
    # (Optional) Continue that new branch for r>r_switch
    # ...
    
    # Plot an example cross-section
    plt.figure()
    plt.plot(x, u_new_branch, 'b-', label=f'New Branch at r={r_switch:.3f}')
    plt.plot(x, u_star_sol, 'r--', label=f'Old Branch at r={r_star:.3f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Branch Switching Demo (Pitchfork PDE)')
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()