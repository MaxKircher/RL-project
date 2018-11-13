import torch
import numpy as np
from scipy.stats import norm

class NN(object):

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

        self.log_dev = torch.ones(self.a_dim, requires_grad=True) #see https://pytorch.org/docs/stable/notes/autograd.html


        # choose action
    def choose_a(self, s):
        mu = self.model(torch.tensor(s)).detach().numpy() # Vektor für den action space
        dev = np.exp(self.log_dev.detach().numpy()) # deviation
        return np.random.normal(mu, dev, 1)


    def q(self, s, a):
        mu = self.model(torch.tensor(s)).detach().numpy() # Vektor für den action space
        dev = np.exp(self.log_dev.detach().numpy()) # deviation
        return norm.pdf(a, mu, dev)

    def ableitbar_q(self, s, a): #liefert das gleiche zurück wie q nur für torch interpretierbar, sodass diese Funktion optmiert werden kann
        mu = self.model(torch.tensor(s)).double()
        dev = torch.exp(self.log_dev).double()
        covariance_matrix = torch.eye(dev.shape[0]).double()*dev
        # factor = 1/(torch.tensor(np.sqrt(2*np.pi))*dev)
        # exponent = -(torch.tensor(a).double()-mu.double()).pow(2)/(2*(dev.double().pow(2)))
        # return  factor*torch.exp(exponent)
        normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix)
        return torch.exp(normal_distribution.log_prob(torch.tensor(a).double()))
