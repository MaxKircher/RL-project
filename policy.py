import torch
import numpy as np

class NN(object):

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        inter_dim = 10
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.s_dim, inter_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(inter_dim, self.a_dim),
        )

        self.log_dev = torch.ones(self.a_dim, requires_grad=True)



    def choose_a(self, s):
        mu = self.model(torch.tensor(s)).detach().numpy()
        dev = np.exp(self.log_dev.detach().numpy())
        return np.random.normal(mu, dev, self.a_dim)
