import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) Define the System and Its Derivatives
###############################################################################
def F(x, r):
    """Right-hand side of the ODE: x' = r*x - x^3."""
    return r*x - x**3

def dF_dx(x, r):
    """Jacobian wrt x: dF/dx = r - 3*x^2."""
    return r - 3*x**2

def d2F_dx2(x, r):
    """Second derivative wrt x: d^2F/dx^2 = -6*x."""
    return -6*x

def d3F_dx3(x, r):
    """Third derivative wrt x: d^3F/dx^3 = -6 (constant)."""
    return -6

def dF_dr(x):
    """Partial derivative of F wrt r: dF/dr = x."""
    return x


###############################################################################
# 2) Locate Bifurcation by Simple Continuation
###############################################################################
def continuation_trivial_branch(r_min=-1.0, r_max=1.0, steps=50):
    """
    Continuation of the trivial solution x=0 from r_min to r_max.
    For this toy problem, x=0 is always a solution if r*x - x^3=0 => x=0.
    We'll track that solution and detect the sign of the Jacobian.
    """
    rs = np.linspace(r_min, r_max, steps)
    xs = np.zeros_like(rs)  # trivial solution x=0
    # We'll check when dF/dx = r - 3*x^2 crosses zero => r=0 for x=0
    # So the pitchfork is obviously at r=0, x=0 in this example.
    return rs, xs

def detect_bifurcation(rs, xs):
    """
    Look for an r where the real eigenvalue crosses zero.
    For x=0 => dF/dx = r. The crossing is at r=0.
    Return (x*, r*) = (0,0).
    """
    # In a real PDE/ODE, you'd look for sign changes in eigenvalues.
    # Here, we know r=0 is the pitchfork.
    # We'll do a simple sign check for demonstration.
    for i in range(len(rs)-1):
        if rs[i]*rs[i+1] < 0:
            # found sign change near 0
            return 0.0, 0.0
    # If we didn't find it, just return None
    return None, None


###############################################################################
# 3) Normal Form Analysis at the Pitchfork
###############################################################################
def normal_form_analysis(x_star, r_star):
    """
    Compute:
      - null vector phi (satisfies J(x_star,r_star)*phi=0)
      - adjoint vector psi (satisfies J^T*psi=0)
      - normal form coefficients alpha, beta, sigma
    For a 1D system, phi, psi are just scalars (or we can treat them as 1D vectors).
    """
    # 3.1: Jacobian at the bifurcation
    J_star = dF_dx(x_star, r_star)  # scalar
    # Solve J_star * phi = 0 => r_star - 3*x_star^2=0
    # For x_star=0, r_star=0 => J_star=0 => any phi is an eigenvector
    # We'll pick phi=1 for convenience.
    phi = 1.0

    # 3.2: Solve J_star^T * psi=0 => same as J_star=0 => pick psi=1
    # We then want <psi, phi> = 1 => 1*1=1 => that's fine.
    psi = 1.0

    # 3.3: Compute second derivative term alpha = <psi, D^2F(x_star)[phi, phi]>
    # In 1D, D^2F is scalar, so alpha = D^2F(x_star)*phi^2
    alpha = d2F_dx2(x_star, r_star)*(phi**2)

    # 3.4: Third derivative term beta = <psi, D^3F(x_star)[phi, phi, phi]>
    # In 1D, beta = D^3F(x_star)*phi^3
    beta = d3F_dx3(x_star, r_star)*(phi**3)

    # 3.5: Partial wrt r => sigma = <psi, dF/dr(x_star)> * phi ?
    # Actually we want how the linear term depends on (r-r_star).
    # For 1D, typically sigma = dF_dr(x_star)= x_star => but x_star=0 => sigma=0
    # However, let's compute it carefully: sigma = <psi, dF_dr(x_star)> = psi*x_star => 0
    sigma = psi*dF_dr(x_star)  # = 0 in this toy example

    return phi, psi, alpha, beta, sigma


###############################################################################
# 4) Branch Switching Using Normal Form
###############################################################################
def branch_switch(x_star, r_star, phi, beta, eps=1e-2):
    """
    For a pitchfork, if beta>0 => supercritical branch goes up (sqrt(r)),
    if beta<0 => subcritical. We'll pick the sign of r step from sign(beta).
    x_switch = x_star + eps*phi
    r_switch = r_star + sign(beta)*eps
    """
    sign_beta = np.sign(beta) if beta!=0 else 1.0
    x_switch = x_star + eps*phi
    r_switch = r_star + sign_beta*eps
    return x_switch, r_switch


###############################################################################
# 5) Simple Continuation on the New Branch
###############################################################################
def new_branch_continuation(x_init, r_init, steps=30, step_size=0.05):
    """
    We'll do a naive 'Newton' for x such that F(x,r)=0, stepping r from r_init upward.
    Real PDE code would do pseudo-arclength. This is just a demonstration.
    """
    xs = []
    rs = []
    x_current = x_init
    r_current = r_init

    for _ in range(steps):
        # Solve F(x,r_current)=0 for x by Newton in 1D
        x_current = newton_1d(lambda x: F(x, r_current),
                              lambda x: dF_dx(x, r_current),
                              x_current)

        xs.append(x_current)
        rs.append(r_current)

        # Step r
        r_current += step_size
    return np.array(rs), np.array(xs)

def newton_1d(f, df, x0, tol=1e-8, maxiter=50):
    """Simple 1D Newton iteration: x_{n+1} = x_n - f(x_n)/df(x_n)."""
    x = x0
    for _ in range(maxiter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx)<1e-14:
            break
        x_new = x - fx/dfx
        if abs(x_new - x)<tol:
            return x_new
        x = x_new
    return x


###############################################################################
# MAIN DEMO
###############################################################################
def main():
    # 1) Continue trivial solution from r_min to r_max
    r_min, r_max = -1.0, 1.0
    rs_triv, xs_triv = continuation_trivial_branch(r_min, r_max, steps=40)

    # 2) Detect the pitchfork near r=0
    x_star, r_star = detect_bifurcation(rs_triv, xs_triv)
    if x_star is None:
        print("No bifurcation detected in this range.")
        return

    print(f"Bifurcation detected near (x*, r*)=({x_star}, {r_star})")

    # 3) Normal form analysis
    phi, psi, alpha, beta, sigma = normal_form_analysis(x_star, r_star)
    print("Normal Form Analysis at (x*,r*)=", (x_star, r_star))
    print(f"  phi={phi}, psi={psi}, alpha={alpha}, beta={beta}, sigma={sigma}")

    # 4) Branch switching
    eps = 1e-2
    x_switch, r_switch = branch_switch(x_star, r_star, phi, beta, eps=eps)
    print(f"Switching to new branch => initial guess (x, r)=({x_switch}, {r_switch})")

    # 5) Continue the new branch
    r_new, x_new = new_branch_continuation(x_switch, r_switch, steps=30, step_size=0.05)

    # Plot the results
    #  - The trivial branch: (rs_triv, xs_triv=0)
    #  - The new branch: (r_new, x_new)
    plt.figure(figsize=(6,5))
    plt.plot(rs_triv, xs_triv, 'bo-', label='Trivial branch')
    plt.plot(r_new, x_new, 'rx-', label='New branch')
    plt.axhline(0, color='k', ls='--')
    plt.axvline(0, color='k', ls='--')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.title('Pitchfork Bifurcation: Branch Switching Demo')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()