import torch
import numpy as np
from scipy.stats import multivariate_normal

class NN(object):
    '''
        Creates a neural network
        Params:
         s_dim = dimension of the state space
         a_dim = dimension of the action space
    '''
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        inter_dim = 64
        self.model = torch.nn.Sequential(
            torch.nn.Linear(s_dim, inter_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(inter_dim, inter_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(inter_dim, a_dim),
        )
        self.model[-1].weight.data.mul_(0.1)
        self.model[-1].bias.data.mul_(0.0)
        self.model.log_std = torch.nn.Parameter(2.0 * torch.ones(self.a_dim, requires_grad=True))

    '''
        Computes the covariance matrix for the current log_std

        Return:
         covariance_matrix = {numpy ndarray}
    '''
    def get_covariance_matrix(self):
        dev = np.exp(self.model.log_std.detach().numpy())
        covariance_matrix = np.diag(dev)
        return covariance_matrix

    '''
        Chooses a random action from normal distribution, where:
        - mean: computed by NN for given state
        - variance: get_get_covariance_matrix
        
        Params:
         - s: {numpy ndaray} state
    '''
    def choose_action(self, s):
        mu = self.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
        return np.random.multivariate_normal(mu, self.get_covariance_matrix(), 1)

    '''
        Compute probability to choose action a in state s.
        Computation is done in pytorch, so we can perform backward.
        
        Params:
         - s: {numpy ndarray} state
         - a: {numpy ndarray} action
         
         Return:
          - {torch Tensor} probability of state a
    '''
    def pi_theta(self, s, a):
        mu = self.model(torch.tensor(s, dtype = torch.float)).double()
        dev = torch.exp(self.model.log_std).double()
        covariance_matrix = torch.diag(dev)

        normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix)
        return torch.exp(normal_distribution.log_prob(torch.tensor(a).double()))

    '''
        Updates the parameter of the policy
        Parameters:
         - theta_new: {torch Parameters} new parameters (weights and biases) for the network
    '''
    def update_policy_parameter(self, theta_new):
        theta_new = theta_new.view(-1)

        self.model.log_std.data = theta_new[:self.a_dim]

        number_of_layers = len(self.model)
        # get right position where we get the params from theta_new:
        j = self.a_dim
        for i in range(number_of_layers):

            if type(self.model[i]) == torch.nn.modules.linear.Linear:
                size_weight = self.model[i].weight.size()
                size_bias = self.model[i].bias.size()

                no_weights = self.model[i].weight.nelement()
                no_bias = self.model[i].bias.nelement()
                # get the new weights
                theta_new_weights = theta_new[j: j + no_weights]
                j += no_weights
                theta_new_bias = theta_new[j: j + no_bias]
                j += no_bias

                self.model[i].weight.data = theta_new_weights.view(size_weight)
                self.model[i].bias.data = theta_new_bias.view(size_bias)

        assert j == theta_new.size(0)

    '''
        Returns parameters of the network
 
        Return:
         - {torch Tensor} parameters of the network
    '''
    def get_parameters(self):
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        theta = torch.zeros(1, number_cols)

        j = 0
        for param in parameters:
            theta[:,j: j + param.nelement()] = param.view(1, -1)
            j += param.nelement()

        return theta

    '''
         Returns gradient of the network.
         backward() has to be performed before

         Return:
          - {torch Tensor} gradients of the network
     '''
    def get_gradients(self):
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        gradient = np.zeros((number_cols, 1))

        j = 0
        for param in parameters:
            gradient[j: j + param.nelement(),:] = param.grad.view(-1, 1)
            j += param.nelement()

        return gradient
