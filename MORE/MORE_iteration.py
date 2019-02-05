import numpy as np
from policy import *
from sample import *
from regression import compute_quadratic_surrogate
from optimization import *
from plot_data import *
from matplotlib import pyplot as plt
import pickle

class More(object):

    def __init__(self, policy, env, N_per_theta, number_of_thetas, memory_size):
        self.sample_generator = Sample(env, policy, N_per_theta, number_of_thetas, memory_size)
        self.policy = policy

    def iterate(self, delta, etha = 1e5, omega = 1):
        d = self.policy.get_number_of_parameters()

        b = np.array(d*[0])
        #b = np.random.randn(d)
        Q = 5*np.eye(d)

        b_history = [b]
        reward_list = np.array([])
        fig = plt.figure()

        count = 0
        while np.absolute(np.diag(Q).sum()) > delta:
            b, Q, rewards, thetas, etha, omega = self.__more_step__(b, Q, etha, omega)

            count += 1
            print("Count: ", count, " Still improving...", np.diag(Q).sum())

            # Plotting
            b_history += [b]
            reward_list = np.append(reward_list, self.sample_generator.sample_single_theta(b))
            plt.figure(fig.number)
            # plt.semilogy()
            plt.plot(range(count), reward_list, c='b')
            plt.show(block=False)
            plt.pause(1e-17)
            plt.savefig("snapshots/bb_rbf.png")

            if (np.mod(count, 30) == 0) or (np.absolute(np.diag(Q).sum()) <= delta):
                plot(rewards, thetas, d, b_history)

            # Save policy in file
            self.policy.set_theta(b)
            dict = {"policy": self.policy}
            with open("policies/bb_rbf.pkl", "wb") as output:
                pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)


    def __more_step__(self, b, Q, etha, omega):
        rewards, thetas, weights = self.sample_generator.sample(b, Q)
        # Weighted Least Squares, weight for a given theta_i is given by pi(theta_i)/pi_i(theta_i) normalized to sum up to 1
        R, r = compute_quadratic_surrogate(thetas, rewards, weights)
        opti = Optimization(Q, b, R, r, .01, 0.99)
        # etha0 = self.__compute_etha0__(1, Q, R)
        x0 = np.asarray([etha, omega])

        sol = opti.SLSQP(x0) #L_BFGS_B(x0) #
        print("Computed etha: {}, omega: {}".format(sol.x[0], sol.x[1]))

        # Update pi
        etha = sol.x[0]
        omega = sol.x[1]
        F = np.linalg.inv(etha * np.linalg.inv(Q) - 2 * R)
        f = etha * np.linalg.inv(Q) @ b + r
        # print("f: ", f)
        # print("F: ", F)

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        #print("parameter change: ", np.abs(b - b_new).sum())
        print("Reward max: ", max(rewards))
        #print("Reward max - min: ", max(rewards) - min(rewards))
        #print("theta = ", b_new)

        return b_new, Q_new, rewards, thetas, etha, omega

    def __compute_etha0__(self, etha0, Q, R):
        F = np.linalg.inv(etha0 * np.linalg.inv(Q) - 2 * R)
        # print("inv(Q): ", np.linalg.inv(Q))
        while not np.all(np.linalg.eigvals(F) > 0):
            #print(etha0)
            etha0 += 1
            F = np.linalg.inv(etha0 * np.linalg.inv(Q) - 2 * R)

        print("etha0 = ", np.linalg.eigvals(F) > 0)

        return etha0


    def my_inv(self, M):
        new = np.zeros(M.shape)
        new[0,0] = M[1,1]
        new[1,0] = -M[1,0]
        new[0,1] = -M[0,1]
        new[1,1] = M[0,0]
        print("1/det: ", 1/np.linalg.det(M))

        return (1/np.linalg.det(M)) * new
