import numpy as np
from scipy.optimize import rosen

'''
    Class that should contains diffrent policies
    Currently:
     - Polynomial policy of degree N
     - Neuronal Network
'''
class Policy(object):

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def set_theta(self, theta):
        raise NotImplementedError("Sublcasses should implement this!")

    def get_action(self, state):
        raise NotImplementedError("Sublcasses should implement this!")

    def get_number_of_parameters(self):
        raise NotImplementedError("Sublcasses should implement this!")

class DebugPolicy(Policy):
    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)

    def set_theta(self, x):
        #return 1.9
        #return np.sin(x)
        return -x * (x - 20) * (x - 2) * (x + 23) / 1000

    def get_number_of_parameters(self):
        return 1

# 
# class Rosenbrock(DebugPolicy):
#     def __init__(self, state_dim, action_dim):
#         Policy.__init__(self, state_dim, action_dim)
#
#     def set_theta(self, thetas):
#         return -rosen(thetas)[0]
#
#     # Number of parameters / dimension of rosenbrock can be changed arbitrarily here:
#     def get_number_of_parameters(self):
#         return 1
