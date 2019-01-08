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

        # Compute the coloumns of the Jacobi-Matrix
        number_cols = sum(p.numel() for p in policy_net.parameters())

        ''' TODO: Ist Jacobi_matrix[0,0] = 0 ??? '''
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
            j = 1
            for theta in thetas:
                grad = theta.grad.view(-1)
                Jacobi_matrix[0,j:j + grad.size(0)] += grad
                j += grad.size(0) # see TODO: Hopefully the first entry of the first row is 0

            assert j == Jacobi_matrix.shape[1] + 1

            # Add the derivatives for the log_std which are ones because std is a theta-param
        assert (mu_actions.size(1) == self.policy.model.log_std.size(0)) , "dimensions have to match"
        Jacobi_matrix[1, 0] = 1

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
    def line_search(self, beta, delta, search_direction, theta_old, states, actions, Q):

        # For debugging:
        theta_old_sum = self.policy.get_parameter_as_tensor().detach().numpy().sum()

        old_obj = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        log_std_old = self.policy.model.log_std.detach().numpy()
        mean_old = self.policy.model(torch.tensor(states, dtype = torch.float)).detach().numpy()

        for i in range(1, 100):
            # print(beta.shape , " ", theta_old.view(-1).size(), " ", torch.tensor(search_direction, dtype = torch.float).size())

            # Save updated for policy parameter in variable theta_new
            theta_new = theta_old + beta * torch.tensor(search_direction.T, dtype = torch.float)

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


    def kl_normal_distribution_torch(self, mu_new, log_std_new):
        mu_old = torch.autograd.Variable(mu_new.detach())
        log_std_old = torch.autograd.Variable(log_std_new.detach())
        var_new = torch.pow(torch.exp(log_std_new), 2)
        var_old = torch.pow(torch.exp(log_std_old), 2)
        #print((mu_new - mu_old).sum(), (var_new - var_old).sum())

        temp = (mu_old - mu_new) * (mu_old - mu_new)
        mean_grad = torch.autograd.grad(temp[0], self.policy.model.parameters(), create_graph=True, allow_unused=True)
        print("mean grad: ", mean_grad)


        kl = log_std_new - log_std_old + (var_old - torch.pow(mu_old - mu_new, 2)) / (2.0 * var_new) -0.5
        # average over samples, sum over action dim
        kl = kl.mean(0).sum(0)
        #print("kl: ", kl)
        return kl

    # Das Innere von Formel (14) (hier machen wir empirischen Eerwartungswert)
    def objective_theta(self, pi_theta, states, actions, Q):
        sum = torch.zeros(1).double()
        #sum_el = torch.zeros(1).double()

        for i in range(actions.shape[0]):
            s = states[i]
            a = actions[i]
            #sum_el += torch.exp(torch.log(pi_theta(s, a)) - torch.log(torch.tensor(self.policy.q(s, a)).double())) * Q[i]
            sum += pi_theta(s, a) * Q[i] / torch.tensor(self.policy.q(s, a)).double()
        #print("el: ", sum_el, " without: ", sum)
        return sum/actions.shape[0]

    def compute_objective_gradients(self, states, actions, Q):
        self.policy.model.zero_grad()
        to_opt = self.objective_theta(self.policy.pi_theta, states, actions, Q)
        to_opt.backward()

        g = self.policy.get_gradients_as_tensor()
        #print("g.shape = ", g.shape)
        return g


    def beta(self, delta, s, JM, FIM):
        JM_s = JM @ s
        FIM_JM_s = FIM @ JM_s
        JMT_FIM_JM_s = JM.T @ FIM_JM_s

        #print(s.T @ JMT_FIM_JM_s)

        return np.power((2*delta)/(s.T @ JMT_FIM_JM_s)[0,0], 0.5)


    def beta_fvp(self, states, delta, s):
        fvp_s = self.fvp(states, s).detach().numpy()
        return np.power((2*delta)/(s.T @ fvp_s)[0,0], 0.5)



    '''
        Compute fisher vector product with x
    '''
    def fvp(self, states, x):
        self.policy.model.zero_grad()
        mean = self.policy.model(torch.tensor(states, dtype = torch.float, requires_grad=True))
        #mean_grad = torch.autograd.grad(mean[0], self.policy.model.parameters(), create_graph=True, allow_unused=True)
        #print("mean grad: ", mean_grad)
        log_std = self.policy.model.log_std

        kl = self.kl_normal_distribution_torch(mean, log_std)


        grad = torch.autograd.grad(kl, self.policy.model.parameters(), create_graph=True)
        grad = torch.cat([g.view(-1,1) for g in grad])

        # kl.backward(retain_graph=True)
        # grad = self.policy.get_gradients_as_tensor().transpose(0,1)

        print("grad: ", grad)
        print("x: ", x.size())
        grad_x = grad * torch.autograd.Variable(x)
        print("grad_x: ", grad_x.size())
        grad_x = grad_x.sum()
        print("grad_x: ", grad_x.size())





        grad = torch.autograd.grad(grad_x, self.policy.model.parameters())
        grad = torch.cat([g.contiguous().view(-1,1) for g in grad]).detach()

        #self.policy.model.zero_grad()
        # grad_x.backward(create_graph=False)
        # grad = self.policy.get_gradients_as_tensor()

        print("fvp: ", grad.size())

        return grad
