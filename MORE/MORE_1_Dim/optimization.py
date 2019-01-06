import numpy as np
from scipy.optimize import minimize


'''
    Use: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
         method="SLSQP" https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Variables have to be in numpy
    We need to pass a starting point, I suggest x = (1,1) (could be discussed with Boris)
'''

class Optimization(object):
    def __init__(self, Q, b, R, r, epsilon, gamma):
        self.Q = Q
        self.b = b
        self.R = R
        self.r = r
        self.epsilon = epsilon
        self.beta = self.compute_beta(gamma, Q)

    '''
        Our function g, which we want to minimize
        Param x: contains the langragian parameter etha and omega
    '''
    def objective(self, x):
        etha = x[0]
        omega = x[1]

        F = 1/(etha/(self.Q) - 2 * self.R)
        f = etha/(self.Q) * self.b + self.r

        # print("Log warning 1: ", np.linalg.det(self.Q))
        A = 2 * np.pi * (etha + omega) * F
        if A < 0:
            print("Log warning 2: ", A)
        # Our objective Function g, see Chapter 2.2
        g = etha * self.epsilon - self.beta * omega + .5 * (f * F * f \
                - etha * self.b *  1/(self.Q) * self.b \
                - etha * np.log((2 * np.pi * self.Q)) \
                + (etha + omega) * np.log(A))

        return g

    '''
        F needs to be positive definite (all evals of F > 0)
    '''
    def constraint(self, x):
        etha = x[0]
        F = 1 / (etha / self.Q - 2 * self.R)
        return F - 1e-2

    def SLSQP(self, x0):
        bnds = ((x0[0], None), (1e-5, None))
        cons = {'type': 'ineq', 'fun': self.constraint}
        soloution = minimize(self.objective, x0, method = 'SLSQP', bounds = bnds, constraints = cons)
        return soloution

    def L_BFGS_B(self, x0):
        bnds = ((x0[0], None), (1e-5, None))
        cons = {'type': 'ineq', 'fun': self.constraint}
        soloution = minimize(self.objective, x0, method = 'L-BFGS-B', bounds = bnds, constraints = cons)
        return soloution

    '''
        Updates our mu and dev for the sampling method
    '''
    def update_pi(self, F, f, etha, omega):
        return F * f, F * (etha + omega)

    '''
        H[pi0] = -75
        q = actual probability distribution
        we use: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
    '''
    def compute_beta(self, gamma, Q):
        k = 1
        H_q = (k/2) + (k * np.log(2 * np.pi)) / 2 + np.log(Q) / 2
        print("H_q: ", H_q)
        # TODO: H_pi0 übergeben
        H_pi0 = -75
        beta = gamma * (H_q - H_pi0) + H_pi0

        return beta
