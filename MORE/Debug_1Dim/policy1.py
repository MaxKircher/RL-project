import numpy as np
# import torch

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

    def set_theta(self, thetas):
        return 1.9
        #return -thetas * (thetas - 10) * (thetas - 2) * (thetas + 13) / 1000

    def get_number_of_parameters(self):
        return 1
