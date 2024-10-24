from New_Rap import New_Rap
import numpy as np
from scipy.optimize import fsolve

# Solve a system of nonlinear equations via Newton Raphson Method
# Faster convergence than "fsolve" from scipy.optimize library
# Comparison its showing below

# index: useful to set the number of outputs
#     1: X found
#     2: || and status (1: converged // 0: iterations limited // -1: did not converge)
#     3: || and iterations
#     4: || and jacobian matrix at the last iteration

# progress:
#     1: shows a chart to visualize method behavior
#     0: do not show any chart

# Defining the system of equations
def fun(X):

    G = np.empty([4])

    G[0] = X[0]**2 + X[1]**2 + X[2] - 1
    G[1] = X[0] + X[1]**2 - X[2]**2 + X[3] - 1
    G[2] = X[0]**3 - X[1] + X[3]**2
    G[3] = X[0]**2 - X[1]**2 + X[2] - X[3]

    return G


X0 = np.array([0.5, 0.5, 0.5, 0.5])  # Initial guess
tol = 1e-10                          # Error tolerance
max_iter = 30                        # Iterations limit

# Building system framework (G(X) = 0)
sys = New_Rap(fun, X0, tol, max_iter)

# Calling the solver
X, exitflag, iter = sys.solve_sys(index=3, progress=1)

# Compare with "fsolve"
res = fsolve(fun, X0, full_output=True)

print('\n Our algorithm ------> ', X)
print('\n fsolve -------> ', res)
