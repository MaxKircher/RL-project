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
        policy_id:          String to handle diffrent policies

        Returns:
         - rewards: Is a list where the i-th entry corresponds to the average reward of the i-th theta_i
         - thetas:  Is a list where the i-th entry is a random value returned from the multivariate Gaussian
    '''
    def sample(self, N_per_theta, number_of_thetas, pi, mu, dev, policy_id):
        rewards = []
        thetas = []

        for j in range(number_of_thetas):
            # theta is a numpy matrix and needs to be transformed in desired list format
            theta = pi(mu, dev)
            reward = 0
            s = self.env.reset()

            for i in range(N_per_theta):
                if policy_id == "polynomial_policy":
                    # transforms theta into the desired list format
                    theta_transformed = self.theta_as_list(theta, self.state_dim)

                    # Preprocessing for the second loop
                    self.policy.set_theta(theta_transformed)
                    a = self.policy.polynomial_policy(s)
                elif policy_id == "nn_policy":
                    a = self.policy.nn_model(torch.tensor(s)).detach().numpy()
                else:
                    print("invalid policy_id")
                    return None

            s, r, d, i = self.env.step(np.asarray(a))
            reward += r
            if d:
                s = self.env.reset()

            avg_reward = reward / N_per_theta
            rewards += [avg_reward]
            thetas += [theta]
        print("Sampling successfull")
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
