import torch
import numpy as np
from scipy.stats import multivariate_normal

class NN(object):
    def __init__(self, s_dim, a_dim):
        '''
            Creates a neural network
            :param s_dim: {int} dimension of the state space
            :param a_dim: {int} dimension of the action space
        '''
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
        self.model.log_std = torch.nn.Parameter(-0.1 * torch.ones(self.a_dim, requires_grad=True))

    def get_covariance_matrix_numpy(self):
        '''
            Computes the covariance matrix for the current log_std

            :return: {numpy ndarray} covariance_matrix
        '''
        dev = np.exp(self.model.log_std.detach().numpy())
        covariance_matrix = np.diag(dev)
        return covariance_matrix


    def choose_a(self, s):
        '''
        Choose a action for given state
        :param s: {numpy ndarray} state
        :return: {numpy ndarray} action
        '''
        mu = self.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
        return np.random.multivariate_normal(mu, self.get_covariance_matrix_numpy(), 1)


    def q(self, s, a):
        '''
        Compute probability to choose action a in state s
        :param s: {numpy ndarray} state
        :param a: {numpy ndarray} action
        :return: {float} probability
        '''
        mu = self.model(torch.tensor(s, dtype = torch.float)).detach().numpy() # Vektor für den action space
        return multivariate_normal.pdf(a, mu, self.get_covariance_matrix_numpy())

    def pi_theta(self, s, a):
        '''
        Compute probability to choose action a in state s. Use pytorch to be able to compute gradient.
        :param s: {numpy ndarray} state
        :param a: {numpy ndarray} action
        :return: {torch tensor} probability
        '''
        mu = self.model(torch.tensor(s, dtype = torch.float)).double()
        dev = torch.exp(self.model.log_std).double()
        covariance_matrix = torch.diag(dev)

        normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix)
        return torch.exp(normal_distribution.log_prob(torch.tensor(a).double()))

    def update_policy_parameter(self, theta_new):
        '''
        Updates the parameter of the policy
        :param theta_new: {torch tensor} the new policy parameters
        '''
        theta_new = theta_new.view(-1)

        # keine negativen Varianzen, da wir den logarithmus speichern
        self.model.log_std.data = theta_new[:self.a_dim]

        # split parameter for the desired model
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

    def get_parameter_as_tensor(self):
        '''
        Get the parameters of the ploicy
        :return: {torch tensor} parameters
        '''
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        theta = torch.zeros(1, number_cols)

        j = 0
        for param in parameters:
            theta[:,j: j + param.nelement()] = param.view(1, -1)
            j += param.nelement()

        return theta

    def get_gradients(self):
        '''
        Returns the gradients with respect to all parameters.
        Note, that backward has to be performed before.
        :return: {numpy ndarray} gradients
        '''
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        gradient = np.zeros((number_cols, 1))

        j = 0
        for param in parameters:
            gradient[j: j + param.nelement(),:] = param.grad.view(-1, 1)
            j += param.nelement()

        return gradient
