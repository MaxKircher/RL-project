import torch
import numpy as np
import pickle

from network import NN

class Policy(NN):
    def __init__(self, in_dim, out_dim, interdims):
        '''
        Creates a neural network
        :param in_dim: dimension of the state space
        :param out_dim: dimension of the action space
        :param interdims: {list of int} dimensions of intermediate layers
        '''
        NN.__init__(self, in_dim, out_dim, interdims)
        #self.model[-1].weight.data.mul_(0.1)
        #self.model[-1].bias.data.mul_(0.0)
        self.model.log_std = torch.nn.Parameter(2.4 * torch.ones(self.out_dim, requires_grad=True))


    def get_covariance_matrix(self):
        '''
        Computes the covariance matrix for the current logarithm of the standard deviation
        :return: {numpy ndarray} covariance matrix
        '''
        dev = np.exp(self.model.log_std.detach().numpy())
        covariance_matrix = np.diag(dev)
        return covariance_matrix

    def choose_action(self, s):
        '''
        Chooses a random action from this normal distribution, where:
         - mean: The mean is computed by our NN for a given state
         - variance: get_covariance_matrix
        :param s: {numpy ndaray} state
        :return: {numpy ndarray} action
        '''
        mu = self.model(torch.tensor(s, dtype=torch.float)).detach().numpy()
        return np.random.multivariate_normal(mu, self.get_covariance_matrix(), 1)

    def pi_theta(self, s, a):
        '''
        Computes the probability of choosing action a in state s
        :param s: {numpy ndarray} state
        :param a: {numpy ndarray} action
        :return: {torch Tensor} probability of action a
        '''
        mu = self.model(torch.tensor(s, dtype = torch.float)).double()
        dev = torch.exp(self.model.log_std).double()
        covariance_matrix = torch.diag(dev)

        normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix)
        return torch.exp(normal_distribution.log_prob(torch.tensor(a).double()))

    def update_parameter(self, theta_new):
        '''
        Updates the parameter of the policy
        :param theta_new: {torch Parameters} new parameters (weights and biases) for the network
        :return: None
        '''
        theta_new = theta_new.view(-1)

        self.model.log_std.data = theta_new[:self.out_dim]
        print("log std = ", theta_new[:self.out_dim])
        super().update_parameter(theta_new[self.out_dim:])


    def save_model(self, path):
        dict = {"policy": self}
        with open("policies/%s.pkl" %path, "wb+") as output:
            pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)


    def compute_Jacobians(self, states):
        '''
        Computes one Jacobi-Matrix per state sample

        :param states: {numpy ndarray} the sampled states
        :return: {list of numpy matrix} a list of Jacobi matrixes
        '''
        states = torch.tensor(states, dtype = torch.float)
        action_expectations = self.model(states)

        # Compute the coloumns of the Jacobi-Matrix
        number_cols = sum(p.numel() for p in self.model.parameters())

        Jacobi_matrices = []

        # We compute the Jacobi matrix for each state in states
        for i in range(action_expectations.size(0)):
            Jacobi_matrix = np.matrix(np.zeros((self.out_dim * 2, number_cols)))

            # reset gradients:
            self.model.zero_grad()

            for k in range(self.out_dim):
                action_expectations[i,k].backward(retain_graph=True)

                Jacobi_matrix[k,:] = self.get_gradients().T

                Jacobi_matrix[k + self.out_dim, k] = self.model.log_std.exp()[k]
                Jacobi_matrices += [Jacobi_matrix]

        return Jacobi_matrices
