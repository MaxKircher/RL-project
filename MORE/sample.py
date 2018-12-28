import numpy as np
import gym
import quanser_robots
from policy import *
import inspect

class Sample(object):

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
        self.theta_memory = []
        self.reward_memory = []

        # Store the dimension of state and action space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    '''
        sample corresponds to the function pi in the paper. It samples values for theta
        that should optimize our objective function R_theta

        N_per_theta:        Query the env N times with this set of Thetas
        number_of_thetas:   number of theta-sample sets
        L:                  Size of number of memorized thetas/samples
        mu:                 expectation for theta sampling
        dev:                covariance matrix for theta sampling

        Returns:
         - rewards: Is a list where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a list where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, N_per_theta, number_of_thetas, L, mu, dev):
        rewards = []
        thetas = []
        #print("mu, dev: ", mu.shape, dev.shape)

        for j in range(number_of_thetas):
            # theta is a numpy matrix and needs to be transformed in desired list format
            reward = 0
            theta = np.random.multivariate_normal(mu, dev)

            if isinstance(self.policy, DebugPolicy):
                reward = self.policy.set_theta(theta)
            else:
                self.policy.set_theta(theta)
                s = self.env.reset()
                for i in range(N_per_theta):
                    a = self.policy.get_action(s)
                    s, r, d, i = self.env.step(np.asarray(a))
                    reward += r
                    if d:
                        s = self.env.reset()

            if isinstance(self.policy, DebugPolicy):
                avg_reward = reward
                self.reward_memory += [avg_reward]
                self.theta_memory += [theta]
            else:
                avg_reward = reward / N_per_theta
                self.reward_memory += [avg_reward]
                self.theta_memory += [theta]
                
        print("Sampling successfull")
        self.reward_memory = self.reward_memory[-L:]
        self.theta_memory = self.theta_memory[-L:]

        return self.reward_memory, self.theta_memory
