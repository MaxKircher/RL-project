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


    def sample(N_per_theta, number_of_thetas): # this is our function pi
        rewards = []
        for j in range(number_of_thetas):
            theta =  np.random.normal(self.mu, self.dev, 1)
            policy.set_theta(theta)
            reward = 0
            s = env.reset()
            for i in range(N_per_theta):
                a = policy(s)
                s, r, d, i = env.step(a)
                reward += r
            avg_reward = reward / N_per_theta
            rewards += [theta, avg_reward]
        return rewards
