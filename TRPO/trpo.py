import numpy as np
import gym
import quanser_robots
import torch
from policy import *
import copy

class TRPO(object):

    '''
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
    Samples values using the single path method (sp)

    Parameter:
     - policy:  the policy that returns an action
     - s0:      initial state that is the beginning of our sampling sequence
                see 5.1. Single Path
    - T:        Number of samples

    Returns:
     - states:  sampled states beginning with initial state s0
     - actions: sampled actions by passing a state to our policy
     - Q:       state-action value function see page 2 above formula (1)
    '''
    def sample_sp(self, policy, s0, T):
        s = s0
        states = [s0]
        actions = []
        rewards = []
        for i in range(T):
            a = policy.choose_a(s)
            s, r, done, info = self.env.step(a)

            states  += [s]
            actions += [a]
            rewards += [r]

        # Make an array from the lists states and actions
        states = np.array(states)
        actions = np.array(actions)
        Q = np.zeros(T+1)

        # # TODO: Expectation value?!
        for i in range(T-1, -1, -1):
            Q[i] = self.gamma*Q[i+1] + rewards[i]

        return states, actions, Q

###### BEGIN: Appendix C
    '''
        DONE UPDATE MU with stdev

        Computes the Jacobi-Matrix by doing the following steps
            1. Get the corrent policy model (currently one-layer linear NN - should be stochastic?!)
            2. Transform states into tensor for the model
            3. Get mu_actions, a tensor of actions for the states

        Params:
         - states: Is a list of states that contains several single states. A Single
                   state corresponds to x in the paper in chapter C.1

        Returns:
         - Jacobi_matrix: Jacobi-Matrix

        mu_a(x) bestimmen
        aus dem Netzwerk holen und dort auch die Gradientenableitungen
        die Gradienten dann in J Matrix speichern

        1. Wir übergeben pytorch für ein bestimmtes x' mu_a(x')
        2. Dann berechnet pytorch mit backward die Gradienten
        3. Wir speichern das ergebnis als J(x')
        4. Benötigen wir J(x*) für ein anderes x*, so starten wir bei 1. mit x* statt x'
    '''

    # state ist das x im paper
    # x = states
    def compute_Jacobian(self, states):

        # Get the policy model (currently an one layer linear NN)
        policy_net = self.policy.model

        # Transform our prediction to a Tensor
        states = torch.tensor(states, dtype = torch.float)

        # For given state expected action w.r.t. the underlying model
        mu_actions = policy_net(states)

        # Compute the coloumns of the Jacobi-Matrix + pad with size of STDEV
        number_cols = sum(p.numel() for p in policy_net.parameters() + self.policy.stdev.size(0))

        # # TODO: Generalise for multi dimensional actions
        # Anpassen mit stdev
        Jacobi_matrix = np.zeros((mu_actions.size(1) * 2, number_cols))


        # .backward um die Gradienten zu bestimmen (2.)
        print(mu_actions.size())
        avg_grad = 0
        for i in range(mu_actions.size(0)):
            # zero-grad damit die Gradienten zurückgesetzt sind
            policy_net.zero_grad()

            # mehr dim action zweiter Eintrag 0 muss geändert werden
            mu_actions[i,0].backward(retain_graph=True)
            # Abspeichern der thetas = weights
            thetas = list(policy_net.parameters())
            # print("policy_net.parameters() = ", policy_net.parameters())
            # for j in range(4):
            #     print("thetas[0] = ", thetas[j].size())
            # print("len(thetas) = ", len(thetas))
            # Macht man eine for-Schleife drum hat man die Jacobi-Matrix als Lsite aufgeschrieben

            j = 0
            for theta in thetas:
                #print("grad.size = " , theta.grad.size())
                grad = theta.grad.view(-1)
                #print("grad.size.view(-1) = ", grad.size(0))
                #print("grad = ", grad)
                Jacobi_matrix[0,j:j + grad.size(0)] += grad
                j += grad.size(0)

            # Add the derivatives for the stdev which are ones because std is a theta-param
            print("ASSERT: mu_actions.size(1) = self.policy.stdev.size(0) :: " mu_actions.size(1) == self.policy.stdev.size(0))
            eye_matrix = np.eye(self.policy.stdev.size(0))

            # Füge die eye Matrix an die richtige Stelle (ganz unten rechts)
            Jacobi_matrix[self.policy.stdev.size(0):,j:] = eye_matrix

            # for j in range(len(thetas)):
            #     grad = thetas[j].grad
            #     # print("grad = ", grad)
            #     # ... J = Jacobi-Matrix TODO generalisieren für mu_actions.size() > 1
            #     Jacobi_matrix[0,] += grad.reshape(-1)

        return Jacobi_matrix # evtl. noch averagen

    '''
        Computes the Fisher-Information Matrix (FIM)
        We choose the Gaussian-Distribution as our distribution of intrest. Therfore
        by Wiki https://de.wikipedia.org/wiki/Fisher-Information?oldformat=true we obtain
        a simple computable FIM
        Returns:
         - FIM: Fisher Information Matrix w.r.t. mean, i.e. the Matrix M in C.1
    '''
    def compute_FIM_mean(self):
        inverse_vars = self.policy.log_dev.exp().pow(-2).detach().numpy()
        fim = np.eye(self.policy.log_dev.size(0))
        return fim

    '''
    Parameter:
     - beta
     - delta
     - s search direction, i.e. A⁻1 * g
     - thet_old
    '''
    def line_search(self, beta, delta, s, theta_old, states, actions, Q):
        theta_new = theta_old + beta * s

        # Get the policy model (currently an one layer linear NN)
        policy_net = self.policy.model

        # Update the parameters of the model
        policy_theta_new = copy.deepcopy(self.policy)
        policy_theta_new.update_policy_parameter(theta_new)

        loss = loss_theta(policy_theta_new, states, actions, Q)

        '''
            DONE UPDATE MU with stdev
        '''
        # Check if KL-Divergenz is <= delta
        mu_new = torch.zeros(actions.shape[0], 2 * actions.shape[1]) # ist actions.shape[1] definiert?
        mu_old = torch.zeros(actions.shape[0], 2 * actions.shape[1]) # ist actions.shape[1] definiert?

        for i in range(actions.shape[0]):
            s = states[i]
            a = actions[i]
            mu_new[i,:] = policy_theta_new(s, a) # passt das von den Dimensionen?
            mu_old[i,:] = self.policy.q(s, a) # passt das von den Dimensionen?

        # hole covariance_matrix_new/old aus den korrigierten theta_old raus
        delta_threshold = kl_normal_distribution(mu_new, mu_old, covariance_matrix_old, covariance_matrix_new)

        '''
            Computes the KL w.r.t. to a multivariat Gaussian for given normal distri-
            butions mu_old, mu_new and the respective covariance matrices (which are params of the model)

            We compute each summand individually and then add them up to make things easier to debug

            Forumla: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        '''
        def kl_normal_distribution(mu_new, mu_old, covariance_matrix_old, covariance_matrix_new):

            trace = torch.inverse(covariance_matrix_new) * covariance_matrix_old
            scalar_product = (mu_new - mu_old).transpose * torch.inverse(covariance_matrix_new) * (mu_new - mu_old)
            k = mu_new.size(1) # richtiger Eintrag?
            ln = torch.log(torch.det(covariance_matrix_new) / torch.det(covariance_matrix_old))

            delta_threshold = 0.5 * (trace + scalar_product - k + ln)
            print("delta threshold = ", delta_threshold)
            return delta_threshold
###### END: Appendix C

    # Das Innere von Formel (14) (hier machen wir empirischen Eerwartungswert)
    def loss_theta(self, pi_theta, states, actions, Q):
        sum = torch.zeros(1).double()
        #print("trpo.py/loss_theta states.shape[0] = ", states.shape[0])
        print("index range = ", states.shape[0])
        for i in range(actions.shape[0]):
            s = states[i]
            a = actions[i]
            sum += pi_theta(s, a)*Q[i]/torch.tensor(self.policy.q(s, a)).double()
        return sum/states.shape[0] # - damit in compute_loss_gradients maximiert wird (ackward minimirt nämlich)

    '''
        Name optmize = compute_loss_gradients. Because we compute the gradient w.r.t. to the
        loss at theta_old
    '''
    def compute_loss_gradients(self, states, actions, Q):
        self.policy.model.zero_grad()
        to_opt = self.loss_theta(self.policy.pi_theta, states, actions, Q)
        to_opt.backward()
        parameters = list(self.policy.model.parameters())

        number_cols = sum(p.numel() for p in self.policy.model.parameters())
        # print("number_cols" , number_cols)
        g = np.zeros(number_cols)
        j = 0
        for param in parameters:
            grad_param = param.grad.view(-1)
            g[j: j + grad_param.size(0)] = grad_param
            j += grad_param.size(0)
        print("g = ", g.shape)
        return g
########
    def beta(self, delta, s, A):
        return np.power((2*delta)/(s.T*A*s), 0.5)
