import numpy as np
import torch

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

class PolynomialPolicy(Policy):

    def __init__(self, state_dim, action_dim, polynomial_degree):
        Policy.__init__(self, state_dim, action_dim)
        self.polynomial_degree = polynomial_degree

    '''
        thetas:
         - List of matrices
         - theta[i] corresponds to the i-th coefficient of the polynomial, i.e. x^i
         - dim(theta[i]) = array([a_1i, a_2i, ..., a_mi]), where m = state_dim
    '''
    def set_theta(self, thetas):
        self.thetas = self.__theta_as_list__(thetas)


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
    def get_action(self, state):

        # Bias term of the polynomial is the default action
        action = self.thetas[0]

        for i in range(1, self.polynomial_degree + 1):
            action += np.dot(self.thetas[i],np.power(state, i))

        return action

    def get_number_of_parameters(self):
        return 1 + self.polynomial_degree * self.state_dim

    '''
        TODO:
         - Generalize w.r.t. state dimension

        Transforms theta which is a numpy array into a list to compute the dot product
        in the function (see policy.py) polynomial_policy

        Returns:
         - list of the format [a_0, array([a_11, a_21, ..., a_m1]), ..., array([a_1n, a_2n, ..., a_mn])]
            - a_0 =         Bias term of the polyonomial
            - a_1-vector =  array([a_11, a_21, ..., a_m1]) the coefficient of x¹
            - a_n-vector =  array([a_1n, a_2n, ..., a_mn]) the coefficient of x^n
            - m = state_dimension
    '''
    def __theta_as_list__(self, theta):

        list = [theta[0]]
        T = (theta.shape[0] - 1) / self.state_dim

        for i in range(int(T)):
            list += [np.array(theta[self.state_dim * i + 1 : self.state_dim * (i + 1) + 1])]

        return list


class NeuronalNetworkPolicy(Policy):

    def __init__(self, state_dim, action_dim):
        Policy.__init__(self, state_dim, action_dim)
        # Wir brauchen keine thetas, da wir das NN irgendwie initialisieren

        inter_dim_1 = 5
        inter_dim_2 = 5
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, inter_dim_1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(inter_dim_1, inter_dim_2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(inter_dim_2, self.action_dim),
        )

    def get_action(self, state):
        action = self.nn_model(torch.tensor(state)).detach().numpy()
        return action

    def set_theta(self, theta):
        theta = torch.tensor(theta).float()
        theta = theta.view(-1)
        # print(theta.size())

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
