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
    def compute_Jacobian(self, states):

        # Get the policy model (currently an one layer linear NN)
        policy_net = self.policy.model

        # Transform our states to a Tensor
        states = torch.tensor(states, dtype = torch.float)

        # For given state expected action w.r.t. the underlying model
        mu_actions = policy_net(states)

        # Compute the coloumns of the Jacobi-Matrix + pad with size of log_dev
        number_cols = sum(p.numel() for p in policy_net.parameters())

        ''' TODO: list(Params) ist die Varianz der erste oder letzte Eintrag? '''
        ''' TODO: Ist der entsprechende Gradient auch Null??? '''
        ''' TODO: Wenns korrekt läuft in Policy auslagern '''


        # TODO: Generalise for multi dimensional actions, because of unclearness w.r.t. to M Matrix (the FIM for mu)
        # Two rows for expectation and stdev
        Jacobi_matrix = np.zeros((mu_actions.size(1) * 2, number_cols))

        # We compute gradients for each state in states and then average over the gradients
        avg_grad = 0
        for i in range(mu_actions.size(0)):
            # zero-grad damit die Gradienten zurückgesetzt sind
            policy_net.zero_grad()

            # TODO: actionspace dimension > 1 in Schleife bearbeiten
            mu_actions[i,0].backward(retain_graph=True)

            # Abspeichern der thetas = {weights, biases, stdev}
            thetas = list(policy_net.parameters())

            # Macht man eine for-Schleife drum hat man die Jacobi-Matrix als Lsite aufgeschrieben
            j = 0
            for theta in thetas:
                grad = theta.grad.view(-1)
                Jacobi_matrix[0,j:j + grad.size(0)] += grad
                j += grad.size(0) # see TODO: Hopefully the last entry of the first row is 0

            # Add the derivatives for the log_dev which are ones because std is a theta-param
            assert (mu_actions.size(1) == self.policy.model.log_dev.size(0)) , "dimensions have to match"
            Jacobi_matrix[1, number_cols] = 1 # vs number_cols - 1

        return Jacobi_matrix

    '''
        # TODO: actiospace dimension > 1 (talk to Boris for parametrization of FIM)

        Computes the Fisher-Information Matrix (FIM)
        We choose the Gaussian-Distribution as our distribution of intrest. Therfore
        by Wiki https://de.wikipedia.org/wiki/Fisher-Information?oldformat=true we obtain
        a simple computable FIM

        Returns:
         - FIM: Fisher Information Matrix w.r.t. mean, i.e. the Matrix M in C.1
    '''
    def compute_FIM_mean(self):
        inverse_vars = self.policy.model.log_dev.exp().pow(-2).detach().numpy()
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
        # Get the policy model (currently an one layer linear NN)
        policy_net = self.policy.model

        old_loss = self.loss_theta(self.policy.pi_theta, states, actions, Q)
        covariance_matrix_old = self.policy.get_covariance_matrix_numpy() # TODO: Make sure that we still have the old Params in the model!!!

        for i in range(1, 10):
            # print(beta.shape , " ", theta_old.view(-1).size(), " ", torch.tensor(s, dtype = torch.float).size())

            # Save updated for policy parameter in variable theta_new
            theta_new = theta_old.view(-1) + beta[0,0] * torch.tensor(s, dtype = torch.float)

            # Update the parameters of the model
            policy_theta_new = copy.deepcopy(self.policy)
            policy_theta_new.update_policy_parameter(theta_new)

            '''
                Preprocessing to compute the KL-Divergence
            '''
            mean_new = torch.zeros(actions.shape[0], actions.shape[1]) # ist actions.shape[1] definiert?
            mean_old = torch.zeros(actions.shape[0], actions.shape[1]) # ist actions.shape[1] definiert?

            for k in range(actions.shape[0]):
                state = torch.tensor(states[k], dtype = torch.float)
                # a = actions[i]
                mean_new[k,:] = torch.tensor(policy_theta_new.model(state)) # passt das von den Dimensionen?
                mean_old[k,:] = torch.tensor(self.policy.model(state)) # passt das von den Dimensionen?

            # hole covariance_matrix_new/old aus den korrigierten theta_old raus
            covariance_matrix_new = policy_theta_new.get_covariance_matrix_numpy()
            delta_threshold = self.kl_normal_distribution(mean_new, mean_old, covariance_matrix_old, covariance_matrix_new)

            # Check if KL-Divergenz is <= delta
            if delta_threshold <= delta:
                loss = self.loss_theta(policy_theta_new.pi_theta, states, actions, Q)
                if loss < old_loss:
                    return policy_theta_new
            beta = pow(beta, -i)

        print("Something went wrong!")
        return Null

    '''
        Computes the KL w.r.t. to a multivariat Gaussian for given normal distri-
        butions mu_old, mu_new and the respective covariance matrices (which are params of the model)

        We compute each summand individually and then add them up to make things easier to debug

        Forumla: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    '''
    def kl_normal_distribution(self, mu_new, mu_old, covariance_matrix_old, covariance_matrix_new):

        trace = torch.trace(torch.inverse(covariance_matrix_new) * covariance_matrix_old)

        # print((mu_new - mu_old).size(), " ", torch.inverse(covariance_matrix_new).size(), " ", (mu_new - mu_old).transpose(1,0).size())
        scalar_product = (mu_new - mu_old) * torch.inverse(covariance_matrix_new) * (mu_new - mu_old).transpose(1,0)
        k = mu_new.size(1) # richtiger Eintrag?
        ln = torch.log(torch.det(covariance_matrix_new) / torch.det(covariance_matrix_old))
        # print("det(cov_new) = ", torch.det(covariance_matrix_new))
        # print("det(cov_old) = ",torch.det(covariance_matrix_old))
        print("ln part of KL (to see if covariance matrix are valid) = ", ln) # to determine if covariance matrix entries are valid
        # print("k = mu_new.size(1) = ", k)

        delta_threshold = 0.5 * (trace + torch.trace(scalar_product) / mu_new.size(0) - k + ln)
        print("delta threshold = ", delta_threshold)
        return delta_threshold

    # Das Innere von Formel (14) (hier machen wir empirischen Eerwartungswert)
    def loss_theta(self, pi_theta, states, actions, Q):
        sum = torch.zeros(1).double()

        for i in range(actions.shape[0]):
            s = states[i]
            a = actions[i]
            sum += pi_theta(s, a) * Q[i] / torch.tensor(self.policy.q(s, a)).double()
        return sum/states.shape[0]

    def compute_loss_gradients(self, states, actions, Q):
        self.policy.model.zero_grad()
        to_opt = self.loss_theta(self.policy.pi_theta, states, actions, Q)
        to_opt.backward()
        
        g = self.policy.get_gradients_as_tensor()
        print("g.shape = ", g.shape)
        return g

    def beta(self, delta, s, A):
        return np.power((2*delta)/(s.T*A*s), 0.5)
