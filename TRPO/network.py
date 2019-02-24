import torch
import numpy as np
import pickle

class NN(object):
    def __init__(self, in_dim, out_dim, interdims):
        '''
        Creates a neural network
        :param in_dim: dimension of the state space
        :param out_dim: dimension of the action space
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = torch.nn.Sequential()
        old_dim = in_dim
        for i, dim in enumerate(interdims):
            self.model.add_module("layer %d" %i, torch.nn.Linear(old_dim, dim))
            self.model.add_module("activation %d" %i, torch.nn.Tanh())
            old_dim = dim
        self.model.add_module("output", torch.nn.Linear(old_dim, out_dim))

    def update_parameter(self, theta_new):
        number_of_layers = len(self.model)
        # get right position where we get the params from theta_new:
        j = 0
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

        assert j == theta_new.size(0), (j, " ", theta_new.size(0))

    def get_parameters(self):
        '''
        Returns parameters of the network
        :return: {torch Tensor} parameters of the network
        '''
        return torch.cat([param.view(1, -1) for param in self.model.parameters()], dim=1)


    def get_gradients(self):
        '''
        Returns gradient of the network.
        backward() has to be performed before calling this function
        :return: {numpy ndarray} gradients of the network
        '''

        def get_grad(param):
            if param.grad is None:
                return torch.zeros(param.nelement(), 1)
            else:
                return param.grad.view(-1, 1)
        gradient = torch.cat([get_grad(param) for param in self.model.parameters()], dim=0)
        return gradient.detach().numpy()

