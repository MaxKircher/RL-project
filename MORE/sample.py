import numpy as np
import gym
import quanser_robots
from policy import *

class SAMPLE(object):

    '''
        policy: The current policy for which we want to update Parameter
                 - polynomial
                 - NN
                 - etc
        mu:      Expectation value for multivariate gaussian
        dev:    Standard deviation for multivariate gaussian
    '''
    def __init__(self, env, policy, mu, dev):
        self.env = env
        self.policy = policy
        self.mu = mu
        self.dev = dev

        # Store the dimension of state and action space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    '''
        sample corresponds to the function pi in the paper. It samples values for theta
        that should optimize our objective function R_theta

        N_per_theta:        sample N values for theta_i
        number_of_thetas:   number of unknown parameter theta_i for i = 1, 2, ... number_of_thetas
                            where dim(theta_i) = state_dim

        Returns:
         - rewards: Is a list where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a list where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, N_per_theta, number_of_thetas):
        rewards = []
        thetas = []

        '''
            First loop:
             - For theta_j sample N values for theta (this is done in the second loop)
             - Compute the average reward
             - Store average reward of theta_j in rewards[j]
             - Correspondingly store theta_j in thetas[j]
        '''
        for j in range(number_of_thetas):
            # theta is a numpy matrix and needs to be transformed in desired list format
            theta =  np.random.multivariate_normal(self.mu, self.dev)

            # transforms theta into the desired list format
            theta_transformed = self.theta_as_list(theta, self.state_dim)

            # Preprocessing for the second loop
            self.policy.set_theta(theta_transformed)
            reward = 0
            s = self.env.reset()

            '''
                Second loop:
                 - Sample N values for theta
                 - Compute action a with policy
                 - Update parameter
            '''
            for i in range(N_per_theta):
                a = self.policy.polynomial_policy(s)
                s, r, d, i = self.env.step(a)
                reward += r


            avg_reward = reward / N_per_theta
            rewards += [avg_reward]
            thetas += [theta]
        return rewards, thetas

    '''
        TODO:
         - Generalize w.r.t. state dimension
         - Belongs to polynomial policy because of the special structure

        Transforms theta which is a numpy array into a list to compute the dot product
        in the function (see policy.py) polynomial_policy

        Returns:
         - list of the format [a_0, array([a_11, a_21, ..., a_m1]), ..., array([a_1n, a_2n, ..., a_mn])]
            - a_0 =         Bias term of the polyonomial
            - a_1-vector =  array([a_11, a_21, ..., a_m1]) the coefficient of x¹
            - a_n-vector =  array([a_1n, a_2n, ..., a_mn]) the coefficient of x^n
            - m = state_dimension
    '''
    def theta_as_list(self, theta, state_dim):

        list = [theta[0]]
        T = (theta.shape[0] - 1) / state_dim

        for i in range(int(T)):
            list += [np.array(theta[state_dim * i + 1 : state_dim * (i + 1) + 1])]

        return list
