import numpy as np
import gym
import quanser_robots
from policy import *

class SAMPLE(object):

    def __init__(self, env, policy, mu, dev):
        self.env = env
        self.policy = policy
        self.mu = mu
        self.dev = dev
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]


    def sample(self, N_per_theta, number_of_thetas): # this is our function pi
        rewards = []
        for j in range(number_of_thetas):
            theta =  np.random.multivariate_normal(self.mu, self.dev)  # theta is matrix and needs to be transformed in desired list format
            # HARDCODE
            # theta[4:6] = 0
            # theta[9:11] = 0

            # transform theta to list
            theta_transformed = self.theta_as_list(theta)
            self.policy.set_theta(theta_transformed)
            reward = 0
            s = self.env.reset()
            for i in range(N_per_theta):
                a = self.policy.polynomial_policy(s)
                print("sample.py semple(...): a: ", a)
                s, r, d, i = self.env.step(a)
                reward += r
            avg_reward = reward / N_per_theta
            rewards += [theta_transformed, avg_reward]
        return rewards


    def theta_as_list(self, theta): # TODO state dimensions
        list = [theta[0]]
        T = (theta.shape[0] - 1) / 5
        print("range ", T)
        # TODO generalisieren
        for i in range(int(T)):
            list += [np.array(theta[self.state_dim * i + 1 : self.state_dim * (i + 1) + 1])]

        return list
