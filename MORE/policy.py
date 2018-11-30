import numpy as np
import torch

'''
    Class that should contains diffrent policies
    Currently:
     - Polynomial policy of degree N
     - Neuronal Network
'''
class POLICY(object):

    '''
        polynomial_degree=None:     Only mandatory if we have an polynomial policy
        thetas=None:                Our parameter. Not mandatory.
    '''
    def __init__(self, state_dim, action_dim, polynomial_degree=None, thetas=None): #theta sind unsere Parameter
        self.thetas = thetas
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.polynomial_degree = polynomial_degree

        # init NN
        inter_dim_1 = 10
        # inter_dim_2 = 10
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, inter_dim_1),
            torch.nn.Sigmoid(),
            # torch.nn.Linear(inter_dim_1, inter_dim_2),
            # torch.nn.Sigmoid(),
            torch.nn.Linear(inter_dim_1, self.action_dim),
        )


    '''
        thetas:
         - List of matrices
         - theta[i] corresponds to the i-th coefficient of the polynomial, i.e. x^i
         - dim(theta[i]) = array([a_1i, a_2i, ..., a_mi]), where m = state_dim
    '''
    def set_theta(self, thetas):
        self.thetas = thetas

    '''
    Computes an action w.r.t. the polynomial policy of a degree N

    States:
     - Current state the system has
     - States are the concrete parametrization of our X values in the polynomial

    P(x) = sum_{i=0}^n a_ix^i, where x in R^{state_dim}
     - Power is element wise
     - a_i * x_i is the dot product because of one dimensional action

    Return:
     - action: The concrete computed action when evaluation polynomial(states)
    '''
    def polynomial_policy(self, states):

        # Bias term of the polynomial is the default action
        action = self.thetas[0]

        for i in range(1, self.polynomial_degree + 1):
            action += np.dot(self.thetas[i],np.power(states, i))

        return action

##### For the NN
    def set_theta_NN(self, thetas):
        return None
