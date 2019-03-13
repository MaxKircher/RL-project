import numpy as np
import torch
import copy
from util import kl_normal_distribution

class TRPO(object):

    def __init__(self, policy):
        '''
        Initialize trpo
        :param policy: {NN} the policy
        '''
        self.policy = policy

    def compute_Jacobians(self, states):
        '''
        Compute Jacobi matrices
        :param states: {numpy ndarray} states
        :return: {list of numpy ndarray} Jacobi matrices
        '''
        policy_net = self.policy.model
        a_dim = self.policy.a_dim
        states = torch.tensor(states, dtype = torch.float)
        mu_actions = policy_net(states)

        # Compute the coloumns of the Jacobi-Matrix
        number_cols = sum(p.numel() for p in policy_net.parameters())
        Jacobi_matrices = []

        # We compute gradients for each state in states and then average over the gradients
        for i in range(mu_actions.size(0)):
            Jacobi_matrix = np.matrix(np.zeros((a_dim * 2, number_cols)))

            policy_net.zero_grad()

            for k in range(a_dim):
                mu_actions[i,k].backward(retain_graph=True)

                thetas = list(policy_net.parameters())

                j = 0
                for theta in thetas:
                    grad = theta.grad.view(-1)
                    Jacobi_matrix[k,j:j + grad.size(0)] = grad
                    j += grad.size(0)

                Jacobi_matrix[k + a_dim, k] = self.policy.model.log_std.exp()[k]
                Jacobi_matrices += [Jacobi_matrix]

        return Jacobi_matrices

    def compute_FIM_mean(self):
        '''
        Computes the Fisher-Information Matrix (FIM)
        We choose the Gaussian-Distribution as our distribution of intrest. Therfore
        by Wiki https://de.wikipedia.org/wiki/Fisher-Information?oldformat=true we obtain
        a simple computable FIM

        :return: {numpy ndarray} Fisher Information Matrix w.r.t. mean, i.e. the Matrix M in C.1
        '''
        inverse_vars = self.policy.model.log_std.exp().pow(-2).detach().numpy()
        fim = np.diag(np.append(inverse_vars, 0.5 * np.power(inverse_vars, 2)))
        return fim

    def line_search(self, beta, delta, s, theta_old, states, actions, Q):
        '''
        perform the line search
        :param beta: {float} initial stepsize
        :param delta: {float} KL-constraint
        :param s: {numpy ndarray} step direction, to update parameters
        :param theta_old: {torch tensor} the old parameters
        :param states: {numpy ndarray} sampled states
        :param actions: {numpy ndarray} sampled actions
        :param Q: {numpy ndarray} sampled discounted rewards
        :return: {torch tensor} new policy parameters
        '''
        old_obj = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        log_std_old = self.policy.model.log_std.detach().numpy()
        mean_old = self.policy.model(torch.tensor(states, dtype = torch.float)).detach().numpy()

        for i in range(1, 20):
            theta_new = theta_old + beta * torch.tensor(s.T, dtype = torch.float)

            # Update the parameters of the model
            policy_theta_new = copy.deepcopy(self.policy)
            policy_theta_new.update_policy_parameter(theta_new)

            mean_new = policy_theta_new.model(torch.tensor(states, dtype = torch.float)).detach().numpy()
            log_std_new = policy_theta_new.model.log_std.detach().numpy()

            kl_change = kl_normal_distribution(mean_new, mean_old, log_std_new, log_std_old)

            # Check if KL-Divergenz is <= delta
            if kl_change <= delta:
                obj = self.objective_theta(policy_theta_new.pi_theta, states, actions, Q)
                if obj >= old_obj:
                    improvement = obj-old_obj
                    print("beta = ", beta, "iteration = ", i)
                    print("new objective: ", obj.detach().numpy(), " improved by ", improvement.detach().numpy())
                    return policy_theta_new
            beta = beta * np.exp(-0.5 * i)

        print("Something went wrong!")
        return None


    def objective_theta(self, pi_theta, states, actions, Q):
        '''
        Compute the objective, that shall be optimized
        :param pi_theta: {NN} the new policy
        :param states: {numpy ndarray} sampled states
        :param actions: {numpy ndarray} sampled actions
        :param Q: {numpy ndarray} sampled discounted rewards
        :return: {torch tensor} the value of the objective
        '''
        q = self.policy.pi_theta(states, actions).detach()
        fast_sum = (pi_theta(states, actions) * torch.tensor(Q, dtype=torch.double) / q).sum()

        return fast_sum / actions.shape[0]

    def compute_objective_gradients(self, states, actions, Q):
        '''
        Compute the gradients of the objective
        :param states: {numpy ndarray} sampled states
        :param actions: {numpy ndarray} sampled actions
        :param Q: {numpy ndarray} sampled discounted rewards
        :return: {torch tensor} gradint
        '''
        self.policy.model.zero_grad()
        to_opt = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        to_opt.backward()

        g = self.policy.get_gradients()
        return g

    def beta(self, delta, s, JMs, FIM):
        '''

        :param delta: {float} KL-constraint
        :param s: {numpy ndarray} step direction
        :param JMs: {list of numpy ndarray} Jacobi matrices
        :param FIM: {numpy ndarray} Jacobi matrices
        :return: {float} initial step size
        '''
        sAs = 0
        for JM in JMs:
            sAs += (s.T @ (JM.T @ (FIM @ (JM @ s))))[0,0]
        return np.power((2 * delta) / sAs, 0.5)
