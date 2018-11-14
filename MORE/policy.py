import numpy as np

# contains diffrent policies
class POLICY(object):

    def __init__(self, state_dim, action_dim, degree, theta=None): #theta sind unsere Parameter
        self.theta = theta
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.degree = degree

    def set_theta(theta):
        # shape of theta is list of matrices [[theta_0],[theta_1], ..., [theata_n]] corresponding
        # to coefficients of polynomial
        self.theta = theta

    def polynomial_policy(states): #polynomial policy with polynomial of degree 7
        #init polyonomial
        polynomial = self.theata[0]
        for i in range(1,degree + 1):
            polynomial += self.theta[i]*np.power(states, i)

        return polynomial # is indeed our action
