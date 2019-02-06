import numpy as np
import torch
from scipy.optimize import rosen
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures


class Policy(object):
    '''
    Abstract class that serves as interface for the different policies
    '''

    def __init__(self, state_dim, action_dim):
        '''
        Initialize the policy
        :param state_dim: {int} dimension of state space
        :param action_dim: {int} dimension of action space
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim

    def set_theta(self, theta):
        '''
        Set the parameters of the policy
        :param theta: {numpy ndarray} the new parameters
        :return: None
        '''
        raise NotImplementedError("Sublcasses should implement this!")

    def get_action(self, state):
        '''
        Sample an action for the given state
        :param state: {numpy ndarray} the current state
        :return: {float} the action, that shall be performed
        '''
        raise NotImplementedError("Sublcasses should implement this!")

    def get_number_of_parameters(self):
        '''
        Return the number of parameters
        :return: {int} the number of parameters
        '''
        raise NotImplementedError("Sublcasses should implement this!")

class DebugPolicy(Policy):
    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)

    def set_theta(self, thetas):
        raise NotImplementedError("Sublcasses should implement this!")

    def get_number_of_parameters(self):
        raise NotImplementedError("Sublcasses should implement this!")

class Rosenbrock(DebugPolicy):
    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)

    def set_theta(self, thetas):
        return -rosen(thetas)

    def get_number_of_parameters(self):
        return 2

class Rastrigin(DebugPolicy):
    # https://en.wikipedia.org/wiki/Rastrigin_function
    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)

    def set_theta(self, thetas):
        A = 10
        n = self.get_number_of_parameters()
        return -(A * n + np.sum(thetas**2 - A * np.cos(2 * np.pi * thetas)))

    def get_number_of_parameters(self):
        return 2

class NeuronalNetworkPolicy(Policy):
    '''
    A neural network as policy
    '''

    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)

        inter_dim_1 = 16
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, inter_dim_1),
            torch.nn.Tanh(),
            torch.nn.Linear(inter_dim_1, self.action_dim),
        )

    def get_action(self, state):
        action = self.nn_model(torch.tensor(state, dtype = torch.float)).detach().numpy()
        return action * 50

    def set_theta(self, theta):
        theta = torch.tensor(theta).float()
        theta = theta.view(-1)

        # split parameter for the desired model
        number_of_layers = len(self.nn_model)
        j = 0 # get right position where we get the params from theta
        for i in range(self.action_dim, number_of_layers):

            if type(self.nn_model[i]) == torch.nn.modules.linear.Linear:
                size_weight = self.nn_model[i].weight.size()
                size_bias = self.nn_model[i].bias.size()

                no_weights = self.nn_model[i].weight.nelement()
                no_bias = self.nn_model[i].bias.nelement()
                # get the new weights
                theta_weights = theta[j: j + no_weights]
                j += no_weights
                theta_bias = theta[j: j + no_bias]
                j += no_bias

                self.nn_model[i].weight.data = theta_weights.view(size_weight)
                self.nn_model[i].bias.data = theta_bias.view(size_bias)

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.nn_model.parameters())

class LinearRBF(Policy):
    '''
    RBF features
    '''

    def __init__(self, state_dim, action_dim, number_of_features):
        Policy.__init__(self, state_dim, action_dim)
        # TODO look at this again:
        self.rbf_feature = RBFSampler(gamma=25., n_components=number_of_features)
        self.rbf_feature.fit(np.random.randn(action_dim, state_dim))

    def set_theta(self, theta):
        self.theta = theta

    def get_action(self, state):
        features = self.rbf_feature.transform(state.reshape(1, -1))
        return features @ self.theta[:-self.action_dim] + self.theta[-self.action_dim:]

    def get_number_of_parameters(self):
        return self.rbf_feature.get_params().get("n_components") + self.action_dim

class LinearPolynomial(Policy):
    '''
    Polynomial features
    '''

    def __init__(self, state_dim, action_dim, degree):
        Policy.__init__(self, state_dim, action_dim)
        self.features = PolynomialFeatures(degree)
        self.features.fit(np.ones((1,state_dim)))

    def set_theta(self, theta):
        self.theta = theta

    def get_action(self, state):
        features = self.features.transform(state.reshape(1, -1))
        return features @ self.theta

    def get_number_of_parameters(self):
        return self.features.n_output_features_ #+ self.action_dim
