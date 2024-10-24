import numpy as np
import time
from scipy import linalg as ln
from matplotlib import pyplot as plt


class New_Rap:
    def __init__(self, fun, X0, tol, max_iter):
        self.fun = fun              # system of equations
        self.X0 = X0                # initial guess
        self.tol = tol              # error tolerance
        self.max_iter = max_iter    # max of iterations

# Get the initial value
    def fun_ini_eval(self):
        G = self.fun(self.X0)
        return G

# Get the system dymensions
    def sys_dim(self):
        G = self.fun_ini_eval()
        n_eq = len(G)               # number of equations
        n_var = len(self.X0)        # number of variables

        return n_eq, n_var

# Return the jacobian matrix forall X0
# Numeric derivative method was used
    def jacobian(self, X):
        # Step parameter
        h = 1e-06

        n_eq, n_var = self.sys_dim()
        JAC = np.zeros([n_eq, n_var], dtype=float)
        fcn = self.fun(X)

        for i in range(n_var):
            var_step = np.array(X, dtype=float)
            var_step[i] += h

            # System value within step point
            fcn_plus_h = self.fun(var_step)

            JAC[:, i] = (fcn_plus_h - fcn) / h

        return JAC

    def solve_sys(self, index, progress):
        t_start = time.time()
        X = self.X0
        tol = self.tol
        max_iter = self.max_iter
        ERROR = []
        CONT = []

        # Get the mismatch
        mism = self.fun_ini_eval()

        # Get the jacobian matrix
        JAC = self.jacobian(X)
        JAC_inv = ln.inv(JAC)

        # Update initial guess
        DX = -JAC_inv @ mism
        X = X + DX

        # Update the loop conditions
        error = np.max(np.abs(DX))
        cont = 1
        ERROR.append(error)
        CONT.append(cont)

        while (cont < max_iter) and (error >= tol):
            # Get the mismatch
            mism = self.fun(X)

            # Get jacobian matrix
            JAC = self.jacobian(X)
            JAC_inv = ln.inv(JAC)

            # Update the initial guess
            DX = -JAC_inv @ mism
            X = X + DX

            # Update the loop conditions
            error = np.max(np.abs(DX))
            cont += 1
            ERROR.append(error)
            CONT.append(cont)

        t_end = time.time()
        run_time = t_end - t_start

        if progress == 1:
            plt.figure()
            plt.plot(CONT, ERROR, 'o-b')
            plt.title('Method Progress')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Máx Absolute Error')
            plt.grid()
            plt.show()

        # Converged !
        if cont < max_iter and error <= tol:
            exitflag = 1
            print('\n -------------------- NEWTON RAPHSON METHOD ---------------------------')
            print(' Solution found !')
            print('\n Converged in {:2.5f} sec // N. iter: {}'.format(run_time, cont))

        # Loop limited by number of iterations
        elif cont == max_iter:
            exitflag = 0
            print('Did not converge !')

        # Did not converge
        elif cont == max_iter and error >= tol:
            exitflag = -1

        if index == 1:
            return X
        elif index == 2:
            return X, exitflag
        elif index == 3:
            return X, exitflag, cont
        elif index == 4:
            return X, exitflag, cont, JAC

