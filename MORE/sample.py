import numpy as np
import gym
import quanser_robots
from policy import *
import inspect
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

class Sample(object):
    '''
        N_per_theta:        Query the env N times with this set of Thetas
        number_of_thetas:   number of theta-sample sets
        memory_size:        Number of memorized thetas/samples
    '''
    def __init__(self, env, policy, N_per_theta, number_of_thetas, memory_size):
        self.env = env
        self.policy = policy
        self.theta_memory = np.zeros((0,policy.get_number_of_parameters()))
        self.reward_memory = np.array([])
        self.log_prob_memory = np.array([])

        self.N_per_theta = int(N_per_theta)
        self.number_of_thetas = number_of_thetas
        self.memory_size = memory_size

    '''
        sample corresponds to the function pi in the paper. It samples values for theta
        that should optimize our objective function R_theta

        mu:                 expectation for theta sampling
        dev:                covariance matrix for theta sampling

        Returns:
         - rewards: Is a array, where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a array, where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, mu, dev):
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

        reward = 0
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
