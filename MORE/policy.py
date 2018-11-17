import numpy as np

'''
    Class that should contains diffrent policies
    Currently:
     - Polynomial policy of degree N
     - TODO: Neuronal Network
'''
class POLICY(object):

    def __init__(self, state_dim, action_dim, degree, theta=None): #theta sind unsere Parameter
        self.theta = theta
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.degree = degree

    def set_theta(self, theta):
        # shape of theta is list of matrices [[theta_0],[theta_1], ..., [theata_n]] corresponding
        # to coefficients of polynomial
        self.theta = theta

    def polynomial_policy(self, states): #polynomial policy with polynomial of degree 7
        #init polyonomial
        polynomial = self.theta[0]
        # print("policy.py polynomial_policy(...): polynomial = " , polynomial)
        for i in range(1, self.degree + 1):
            # print("np.power() ", np.power(states, i))
            # print("dot product: ", np.dot(self.theta[i],np.power(states, i)))
            polynomial += np.dot(self.theta[i],np.power(states, i))
            # print("policy.py polynomial_policy(...): (for loop) polynomial = " , polynomial)

        return polynomial # is indeed our action
