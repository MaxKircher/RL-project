import numpy as np
from policy import *
from sample import *
from regression import compute_quadratic_surrogate
from optimization import *
from plot_data import *
from matplotlib import pyplot as plt
import pickle

class MORE(object):

    def __init__(self, policy, env, N_per_theta, number_of_thetas, memory_size):
        '''
        Initilizes the black box optimizer MORE (i.e. Model Based Relative Entropy-Stochastic-Search)
        :param policy: {Policy} current policy
        :param env: {TimeLimit} current environment
        :param N_per_theta: {int} number of episodes per theta
        :param number_of_thetas: {int} number of thetas to be sampled
        :param memory_size: {int} number of thetas which we like to keep from previous episodes
        '''
        self.sample_generator = Sample(env, policy, N_per_theta, number_of_thetas, memory_size)
        self.policy = policy

    def iterate(self, delta, etha = 1e5, omega = 1):
        '''
        Performs repeatedly a more step through calling __more_step__ until desired abbreviation criterion is fulfilled
        :param delta: {float} threshold to stop the loop
        :param etha: {float} lagragian multiplier of optimization problem
        :param omega: {float} lagragien multiplier of optimization problem
        :return: None
        '''
        d = self.policy.get_number_of_parameters()

        b = np.array(d*[0])
        #b = np.random.randn(d)
        Q = 5*np.eye(d)

        b_history = [b]
        reward_list_mean = np.array([])
        reward_list_var = np.array([])
        fig = plt.figure()

        count = 0
        while np.absolute(np.diag(Q).sum()) > delta:
            b, Q, rewards, thetas, etha, omega = self.__more_step__(b, Q, etha, omega)

            count += 1
            print("Count: ", count, " Still improving...", np.diag(Q).sum())

            # Plot progress
            b_history += [b]
            sampled_rewards = np.array([self.sample_generator.sample_single_theta(b) for i in range(10)])
            reward_list_mean = np.append(reward_list_mean, sampled_rewards.mean())
            reward_list_var = np.append(reward_list_var, sampled_rewards.var())
            plt.figure(fig.number)
            # plt.semilogy()
            plt.plot(range(count), reward_list_mean, c='b')
            plt.show(block=False)
            plt.pause(1e-17)
            plt.savefig("snapshots/BB_rbf30.png")
            file = open("snapshots/BB_rbf30.npy", "wb")
            np.save(file, [reward_list_mean, reward_list_var])
            file.close()

            if (np.mod(count, 50) == 0) or (np.absolute(np.diag(Q).sum()) <= delta):
                plot(rewards, thetas, d, b_history)



            # Save policy in file
            self.policy.set_theta(b)
            dict = {"policy": self.policy}
            with open("policies/BB_rbf30.pkl", "wb") as output:
                pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

        plot(rewards, thetas, d, b_history)


    def __more_step__(self, b, Q, etha, omega):
        '''
        Performes a single MORE step and updates the mean and variance of the search distribution from which we
        sample the policy parameters
        :param b: {numpy ndarray} old mean
        :param Q: {numpy matrix} old covariance matrix
        :param etha: {float} lagragian multiplier of optimization problem
        :param omega: {float} lagragien multiplier of optimization problem
        :return:
         - {numpy ndarray} new mean
         - {numpy matrix} new covariance matrix
         - {numpy ndarray} sampled rewards of the policy with theta values from the old search distribution
         - {numpy ndarray} sampled thetas for the policy from the old search distribution
         - {float} lagragian multiplier of optimization problem after optimization
         - {float} lagragien multiplier of optimization problem after optimization
        '''
        rewards, thetas, weights = self.sample_generator.sample(b, Q)
        # Weighted Least Squares, weight for a given theta_i is given by pi(theta_i)/pi_i(theta_i) normalized to sum up to 1
        R, r = compute_quadratic_surrogate(thetas, rewards, weights)
        opti = Optimization(Q, b, R, r, .01, 0.99)
        x0 = np.asarray([etha, omega])

        sol = opti.SLSQP(x0) #L_BFGS_B(x0) #
        print("Computed etha: {}, omega: {}".format(sol.x[0], sol.x[1]))

        # Update search distribution pi
        etha = sol.x[0]
        omega = sol.x[1]
        F = np.linalg.inv(etha * np.linalg.inv(Q) - 2 * R)
        f = etha * np.linalg.inv(Q) @ b + r

        b_new = F @ f
        Q_new = F * (etha + omega)

        return b_new, Q_new, rewards, thetas, etha, omega