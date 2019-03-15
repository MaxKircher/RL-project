import numpy as np
import gym
import quanser_robots
from policy import *
import inspect
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

class Sample(object):
    def __init__(self, env, policy, N_per_theta, number_of_thetas, memory_size):
        '''
        Initializes our sample generator
        :param env: {TimeLimit} current envirionment
        :param policy: {Policy} current policy
        :param N_per_theta: {int} number of episodes per theta
        :param number_of_thetas: {int} number of thetas to be sampled
        :param memory_size: {int} number of thetas which we like to keep from previous episodes
        '''
        self.env = env
        self.policy = policy
        self.theta_memory = np.zeros((0,policy.get_number_of_parameters()))
        self.reward_memory = np.array([])
        self.log_prob_memory = np.array([])

        self.N_per_theta = int(N_per_theta)
        self.number_of_thetas = number_of_thetas
        self.memory_size = memory_size

    def sample(self, mu, dev):
        '''
        Samples rewards and thetas for specified number of thetas
        :param mu: {numpy ndarray} mean
        :param dev: {numpy ndarray} standard deviation
        :return:
         - {numpy ndarray} sampled rewards
         - {numpy ndarray} sampled thetas
         - {numpy ndarray} weights
        '''
        thetas = np.random.multivariate_normal(mu, dev, self.number_of_thetas)
        rewards = [self.sample_single_theta(thetas[i]) for i in range(thetas.shape[0])]
        log_probs = multivariate_normal.logpdf(thetas, mu, dev)

        self.theta_memory = np.concatenate((self.theta_memory, thetas),0)[-self.memory_size:,:]
        self.reward_memory = np.append(self.reward_memory, rewards)[-self.memory_size:]
        self.log_prob_memory = np.append(self.log_prob_memory, log_probs)[-self.memory_size:]

        new_log_probs = multivariate_normal.logpdf(self.theta_memory, mu, dev)
        weights = np.exp(new_log_probs - self.log_prob_memory)

        return self.reward_memory, self.theta_memory, weights

    def sample_single_theta(self, theta):
        '''
        Samples rewards for a given theta
        :param theta: {numpy ndarray} parameters of the current policy
        :return:
         - {float} average reward
        '''
        reward = 0
        # DebugPolicy is not a actual policy in the reinforcement learning sense,
        # it shall just be optimized:
        if isinstance(self.policy, DebugPolicy):
            reward = self.policy.set_theta(theta)
        else:
            self.policy.set_theta(theta)
            s = self.env.reset()
            episode = 0
            while episode < self.N_per_theta:
                a = self.policy.get_action(s)
                s, r, d, i = self.env.step(np.asarray(a))
                reward += r
                if d:
                    episode += 1
                    s = self.env.reset()
        return reward / self.N_per_theta
