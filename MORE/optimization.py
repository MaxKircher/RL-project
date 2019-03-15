import numpy as np
from scipy.optimize import minimize

class Optimization(object):
    def __init__(self, Q, b, R, r, epsilon, gamma):
        '''
        Initialize the optimizer
        :param Q: {numpy matrix} Covariance matrix, that shall be optimized
        :param b: {numpy ndarray} Mean, that shall be optimized
        :param R: {numpy ndarray} quadratic parameter of surrogate objective
        :param r: {numpy ndarray} linear parameter of surrogate objective
        :param epsilon: {float} upper bound for KL divergence
        :param gamma: {float} discount factor
        '''
        self.Q = Q
        self.b = b
        self.R = R
        self.r = r
        self.epsilon = epsilon
        self.beta = self.compute_beta(gamma, Q)

    def objective(self, x):
        '''
        Our objective function, which we want to minimize
        See MORE paper Chapter 2.2 function g

        :param x: {list of float} contains the langragian multipliers etha and omega
        :return: the value of the objective function
        '''

        etha = x[0]
        omega = x[1]

        # Transform arrays to col-vectors
        self.b = self.b.reshape(self.b.size, 1)
        self.r = self.r.reshape(self.r.size, 1)

        F = np.linalg.inv(etha * np.linalg.inv(self.Q) - 2 * self.R)
        f = etha *  np.linalg.inv(self.Q) @ self.b + self.r

        A = 2 * np.pi * (etha + omega) * F
        detA = np.linalg.det(A)
        if detA < 0:
            print("Log warning 2: ", detA)

        # Our objective Function g, see Chapter 2.2
        g = etha * self.epsilon - self.beta * omega + .5 * (f.T @ F @ f \
                - etha * self.b.T @  np.linalg.inv(self.Q) @ self.b \
                - etha * np.log(np.linalg.det(2 * np.pi * self.Q)) \
                + (etha + omega) * np.log(detA))
        return g[0,0]


    def constraint(self, x):
        '''
        Constrain F to be positive definite (all eigenvalues of F > 0)

        :param x: {list of float} contains the langragian multipliers etha and omega
        :return: {float} greater than 0, if F is positive definite for the given x, negative otherwise
        '''
        etha = x[0]
        F = np.linalg.inv(etha * np.linalg.inv(self.Q) - 2 * self.R)
        evals = np.linalg.eigvals(F) > 0
        return evals.all() - 1e-2

    def SLSQP(self, x0):
        '''
        Optimize the objective
        Use: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
             method="SLSQP" https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        :param x0: {list of float} contains the langragian multipliers etha and omega
        :return: {OptimizeResult} the solution of the optimization
        '''
        bnds = ((1e-5, None), (1e-5, None))
        cons = {'type': 'ineq', 'fun': self.constraint}
        soloution = minimize(self.objective, x0, method = 'SLSQP', bounds = bnds, constraints = cons)
        if not soloution.success:
            print(soloution)
            return self.SLSQP(x0)
        return soloution


    def compute_beta(self, gamma, Q):
        '''
        lower bound on the entropy of the normal distribution corresponding to Q.
        we use: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Differential_entropy

        :param gamma: {float} discount factor
        :param Q: {numpy matrix]} covariance matrix
        :return: {float} lower bound on the entropy
        '''
        H_q = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * Q))
        H_pi0 = -75
        beta = gamma * (H_q - H_pi0) + H_pi0

        return beta
