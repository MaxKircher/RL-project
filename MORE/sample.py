import numpy as np
import gym
import quanser_robots
from policy import *

class SAMPLE(object):

    '''
        > Vererben einbauen!

        policy: The current policy for which we want to update Parameter
                 - polynomial
                 - NN
                 - etc
        Weniger Sinnvoll mu und dev zu setzen, sondern konkret in sample übergeben,
        da sich die immer ändern
        mu: F*f Expectation value for multivariate gaussian
        dev: F(etha + omega) Standard deviation for multivariate gaussian
    '''
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        # self.mu = mu
        # self.dev = dev

        # Store the dimension of state and action space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    '''
        sample corresponds to the function pi in the paper. It samples values for theta
        that should optimize our objective function R_theta

        N_per_theta:        Query the env N times with this set of Thetas
        number_of_thetas:   number of theta-sample sets
        pi:                 The probability distribution to be sampled from

        Returns:
         - rewards: Is a list where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a list where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, N_per_theta, number_of_thetas, pi, mu, dev):
        rewards = []
        thetas = []

        for j in range(number_of_thetas):
            # theta is a numpy matrix and needs to be transformed in desired list format
            theta = pi(mu, dev)
            reward = 0
            s = self.env.reset()

            for i in range(N_per_theta):
                self.policy.set_theta(theta)
                a = self.policy.get_action(s)

            s, r, d, i = self.env.step(np.asarray(a))
            reward += r
            if d:
                s = self.env.reset()

            avg_reward = reward / N_per_theta
            rewards += [avg_reward]
            thetas += [theta]
        print("Sampling successfull")
        return rewards, thetas
