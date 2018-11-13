import torch
import numpy as np

class NN(object):

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        inter_dim = 10
        self.model = torch.nn.Sequential( # neuronale Netzwerk
            torch.nn.Linear(self.s_dim, inter_dim), #Applies a linear transformation to the incoming data
            torch.nn.ReLU(), # Activation functeion, see rectified linear unit
            torch.nn.Linear(inter_dim, self.a_dim),
        )

        self.log_dev = torch.ones(self.a_dim, requires_grad=True) #see https://pytorch.org/docs/stable/notes/autograd.html


        # choose action
    def choose_a(self, s):
        mu = self.model(torch.tensor(s)).detach().numpy() # Vektor für den action space
        dev = np.exp(self.log_dev.detach().numpy()) # deviation
        return np.random.normal(mu, dev, 1)
