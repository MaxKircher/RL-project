import torch
import numpy as np
from scipy.stats import multivariate_normal

class NN(object):

    '''
        Parameter:
            - model:    Linear neural network:
                         - 4 Layers with 1 hidden layer:
                            1. Input-Layer of dimension s_dim
                            2. Intermediate-Layer of dimension inter_dim - (Hidden layer)
                            3. Activation function (counts to layer)
                            4. Outpout-Layer of dimension a_dim
    '''
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        inter_dim = 10
        self.model = torch.nn.Sequential( # neuronale Netzwerk
            torch.nn.Linear(self.s_dim, inter_dim), #Applies a linear transformation to the incoming data
            # inter_dim Knoten im intermediate layer
            torch.nn.ReLU(), # Activation functeion, see rectified linear unit
            torch.nn.Linear(inter_dim, self.a_dim),
        )

        # std = pow(e, lambda) -> lambda = log(std)
        self.model.log_std = torch.nn.Parameter(torch.ones(self.a_dim, requires_grad=True))

    def get_covariance_matrix_numpy(self):
        dev = np.exp(self.model.log_std.detach().numpy())
        covariance_matrix = np.diag(dev) # has dev the right dimension?
        return covariance_matrix

    # choose action
    def choose_a(self, s):
        mu = self.model(torch.tensor(s)).detach().numpy()
        return np.random.multivariate_normal(mu, self.get_covariance_matrix_numpy(), 1)

    # pi_theta old
    def q(self, s, a):
        mu = self.model(torch.tensor(s, dtype = torch.float)).detach().numpy() # Vektor für den action space
        return multivariate_normal.pdf(a, mu, self.get_covariance_matrix_numpy())

    '''
        von pytorch ausgerechnetete Wahrscheinlichkeit die man ableiten kann
        liefert das gleiche zurück wie q nur für torch interpretierbar, sodass diese Funktion optmiert werden kann
    '''
    def pi_theta(self, s, a):
        mu = self.model(torch.tensor(s, dtype = torch.float)).double()
        dev = torch.exp(self.model.log_std).double()
        covariance_matrix = torch.diag(dev)

        # aufstellen normal_distribution
        normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix)
        return torch.exp(normal_distribution.log_prob(torch.tensor(a).double()))

    '''
        Updates the parameter of the policy improve our policy
        Parameter:
         - theta_new: is ideally of the form nn.Parameters otherwise if it's a tensor
                      do nn.Parameters(theta_new)
    '''
    def update_policy_parameter(self, theta_new):

        theta_new = theta_new.view(-1) # we had [1, 73] but 1 was nonsense
        # print(theta_new.size())

        # split parameter for the desired model
        number_of_layers = len(self.model)
        j = 0 # get right position where we get the params from theta_new
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

        # keine negativen Varianzen, da wir den logarithmus speichern
        # print(torch.max(torch.zeros(self.model.log_std.size()),theta_new[j:]))
        self.model.log_std.data = theta_new[j:]

    def get_parameter_as_tensor(self):
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        theta = torch.zeros(self.a_dim, number_cols)

        j = 0
        for param in parameters:
            theta[:,j: j + param.nelement()] = param.view(self.a_dim, -1)
            j += param.nelement()

        return theta

    def get_gradients_as_tensor(self):
        parameters = list(self.model.parameters())
        number_cols = sum(p.numel() for p in self.model.parameters())
        gradient = torch.zeros(self.a_dim, number_cols)

        j = 0
        for param in parameters:
            gradient[:,j: j + param.nelement()] = param.grad.view(self.a_dim, -1)
            j += param.nelement()

        return gradient
