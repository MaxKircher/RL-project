import numpy as np
import gym
import quanser_robots
import torch
from policy import *
import copy

class TRPO(object):

    '''
        shape in numpy und size() in torch

        Parameter:
         - env:     the current environment
         - gamma:   discount factor in (0,1)
         - policy:  the current policy
    '''
    def __init__(self, env, gamma, policy):
        self.env = env
        self.gamma = gamma
        self.policy = policy

    '''
        Computes the Jacobi-Matrix by doing the following steps
            1. Get the corrent policy model
            2. Transform states into tensor for the model
            3. Get mu_actions, a tensor of actions for the states

        Params:
         - states: Is a list of states that contains several single states. A Single
                   state corresponds to x in the paper in chapter C.1

        Returns:
         - Jacobi_matrix: Jacobi-Matrix die nicht geaveraged wird! Da wir nur an der "Richtung" interssiert sind

        mu_a(x) bestimmen
        aus dem Netzwerk holen und dort auch die Gradientenableitungen
        die Gradienten dann in J Matrix speichern

        1. Wir übergeben pytorch für ein bestimmtes x' mu_a(x')
        2. Dann berechnet pytorch mit backward die Gradienten
        3. Wir speichern das ergebnis als J(x')
        4. Benötigen wir J(x*) für ein anderes x*, so starten wir bei 1. mit x* statt x'
    '''
    def compute_Jacobians(self, states):

        # Get the policy model (currently an one layer linear NN)
        policy_net = self.policy.model
        a_dim = self.policy.a_dim

        # Transform our states to a Tensor
        states = torch.tensor(states, dtype = torch.float)

        # For given state expected action w.r.t. the underlying model
        mu_actions = policy_net(states)

        # Compute the coloumns of the Jacobi-Matrix
        number_cols = sum(p.numel() for p in policy_net.parameters())

        # TODO: Generalise for multi dimensional actions, because of unclearness w.r.t. to M Matrix (the FIM for mu)
        # Two rows for expectation and stdev
        Jacobi_matrices = []

        # We compute gradients for each state in states and then average over the gradients
        for i in range(mu_actions.size(0)):
            Jacobi_matrix = np.zeros((a_dim * 2, number_cols))

            # zero-grad damit die Gradienten zurückgesetzt sind
            policy_net.zero_grad()

            # Berechne die Gradienten bzgl. unseres Outputs
            for k in range(a_dim):
                mu_actions[i,k].backward(retain_graph=True) # : korrekt?

                # Abspeichern der thetas = {weights, biases, stdev}
                thetas = list(policy_net.parameters())

                # Macht man eine for-Schleife drum hat man die Jacobi-Matrix als Lsite aufgeschrieben
                j = 0
                for theta in thetas:
                    grad = theta.grad.view(-1)
                    Jacobi_matrix[k,j:j + grad.size(0)] = grad
                    j += grad.size(0) # see TODO: Hopefully the first entry of the first row is 0


                # Add the derivatives for the log_std which are ones because std is a theta-param
                Jacobi_matrix[k + a_dim, k] = self.policy.model.log_std.exp()[k]
                Jacobi_matrices += [Jacobi_matrix]


        return Jacobi_matrices

    '''
        # TODO: action_space dimension > 1 (talk to Boris for parametrization of FIM)

        Computes the Fisher-Information Matrix (FIM)
        We choose the Gaussian-Distribution as our distribution of intrest. Therfore
        by Wiki https://de.wikipedia.org/wiki/Fisher-Information?oldformat=true we obtain
        a simple computable FIM

        Returns:
         - FIM: Fisher Information Matrix w.r.t. mean, i.e. the Matrix M in C.1
    '''
    def compute_FIM_mean(self):
        inverse_vars = self.policy.model.log_std.exp().pow(-2).detach().numpy()
        fim = np.diag(np.append(inverse_vars, 0.5 * np.power(inverse_vars, 2)))
        return fim

    '''
    # TODO: Make sure that self.policy.pi_theta contains old theta values!

    Parameter:
     - beta: Step size
     - delta: KL constraint
     - s: search direction, i.e. A⁻1 * g
     - theta_old: old model parameter
    '''
    def line_search(self, beta, delta, s, theta_old, states, actions, Q):
        # For debugging:
        theta_old_sum = self.policy.get_parameter_as_tensor().detach().numpy().sum()

        old_obj = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        log_std_old = self.policy.model.log_std.detach().numpy()
        mean_old = self.policy.model(torch.tensor(states, dtype = torch.float)).detach().numpy()

        for i in range(1, 100):
            # print(beta.shape , " ", theta_old.view(-1).size(), " ", torch.tensor(s, dtype = torch.float).size())

            # Save updated for policy parameter in variable theta_new
            theta_new = theta_old + beta * torch.tensor(s.T, dtype = torch.float)

            # Update the parameters of the model
            policy_theta_new = copy.deepcopy(self.policy)
            policy_theta_new.update_policy_parameter(theta_new)

            assert self.policy.get_parameter_as_tensor().detach().numpy().sum() == theta_old_sum
            assert policy_theta_new.get_parameter_as_tensor().sum() == theta_new.sum()

            mean_new = policy_theta_new.model(torch.tensor(states, dtype = torch.float)).detach().numpy()
            log_std_new = policy_theta_new.model.log_std.detach().numpy()


            delta_threshold = self.kl_normal_distribution(mean_new, mean_old, log_std_new, log_std_old)

            # Check if KL-Divergenz is <= delta
            if delta_threshold <= delta:
                obj = self.objective_theta(policy_theta_new.pi_theta, states, actions, Q)
                if obj >= old_obj:
                    improvement = obj-old_obj
                    print("beta = ", beta, "iteration = ", i)
                    print("new objective: ", obj.detach().numpy()[0], " improved by ", improvement.detach().numpy()[0])
                    return policy_theta_new
            beta = beta * np.exp(-0.5 * i) # beta / 2 # How to reduce beta?

        print("Something went wrong!")
        return None


    """
        Use, that only variances are given -> dimensions are independant
    """
    def kl_normal_distribution(self, mu_new, mu_old, log_std_new, log_std_old):
        var_new = np.power(np.exp(log_std_new), 2)
        var_old = np.power(np.exp(log_std_old), 2)
        #print((mu_new - mu_old).sum(), (var_new - var_old).sum())

        kl = log_std_new - log_std_old + (var_old - np.power(mu_old - mu_new, 2)) / (2.0 * var_new) -0.5
        # average over samples, sum over action dim
        kl = np.abs(kl.mean(0)).sum(0)
        #print("kl: ", kl)
        return kl

    # Das Innere von Formel (14) (hier machen wir empirischen Eerwartungswert)
    def objective_theta(self, pi_theta, states, actions, Q):
        sum = torch.zeros(1).double()

        for i in range(actions.shape[0]):
            s = states[i]
            a = actions[i]
            sum += pi_theta(s, a) * Q[i] / torch.tensor(self.policy.q(s, a)).double()
        return sum/actions.shape[0]

    def compute_objective_gradients(self, states, actions, Q):
        self.policy.model.zero_grad()
        to_opt = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        to_opt.backward()

        g = self.policy.get_gradients_as_tensor()
        #print("g.shape = ", g.shape)
        return g


    def beta(self, delta, s, A):
        return np.power((2*delta)/(s.T @ A @ s)[0,0], 0.5)