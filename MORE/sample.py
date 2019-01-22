import numpy as np
import gym
import quanser_robots
from policy import *
import inspect

class Sample(object):
    '''
        N_per_theta:        Query the env N times with this set of Thetas
        number_of_thetas:   number of theta-sample sets
        memory_size:        Number of memorized thetas/samples
    '''
    def __init__(self, env, policy, N_per_theta, number_of_thetas, memory_size):
        self.env = env
        self.policy = policy
        self.theta_memory = []
        self.reward_memory = []

        self.N_per_theta = N_per_theta
        self.number_of_thetas = number_of_thetas
        self.memory_size = memory_size

        # Store the dimension of state and action space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    '''
        sample corresponds to the function pi in the paper. It samples values for theta
        that should optimize our objective function R_theta

        mu:                 expectation for theta sampling
        dev:                covariance matrix for theta sampling

        Returns:
         - rewards: Is a list where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a list where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, mu, dev):
        rewards = []
        thetas = []

        for j in range(self.number_of_thetas):
            reward = 0
            theta = np.random.multivariate_normal(mu, dev)

            if isinstance(self.policy, DebugPolicy):
                reward = self.policy.set_theta(theta)
            else:
                self.policy.set_theta(theta)
                s = self.env.reset()
                for i in range(self.N_per_theta):
                    a = self.policy.get_action(s)
                    s, r, d, i = self.env.step(np.asarray(a))
                    reward += r * 1e5 # TODO: Als Skalierungsfaktor übergeben?
                    if d:
                        s = self.env.reset()

            if isinstance(self.policy, DebugPolicy):
                avg_reward = reward
                self.reward_memory += [avg_reward]
                self.theta_memory += [theta]
            else:
                avg_reward = reward / self.N_per_theta
                self.reward_memory += [avg_reward]
                self.theta_memory += [theta]

        print("Sampling successfull")
        self.reward_memory = self.reward_memory[-self.memory_size:]
        self.theta_memory = self.theta_memory[-self.memory_size:]

        return self.reward_memory, self.theta_memory

    def training_sample(self, number_of_thetas, mu, dev, ts):
        rewards = []
        thetas = []

        for j in range(self.number_of_thetas):
            reward = 0
            theta = np.random.multivariate_normal(mu, dev)

            self.policy.set_theta(theta)
            for j in range(len(ts)):
                state = ts[j]
                a = self.policy.get_action(state)
                s, r, d, i = self.env.step(np.asarray(a))
                reward += r

            rewards += [reward]
            thetas += [theta]

        print("Sampling successfull")

        return rewards, thetas
