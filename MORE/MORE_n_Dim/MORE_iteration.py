import numpy as np
from policy import *
from sample import *
from regression import * # , X
from optimization import *
from plot_data import *

class More(object):

    def __init__(self, delta, policy, env, N_per_theta, number_of_thetas, memory_size):
        self.delta = delta
        self.sample_generator = Sample(env, policy, N_per_theta, number_of_thetas, memory_size)
        self.policy = policy
        self.N_per_theta = N_per_theta
        self.number_of_thetas = number_of_thetas
        self.memory_size = memory_size

    def iterate(self):
        #### Raus in die test, cf. TODO
        d = self.policy.get_number_of_parameters()
        #b = np.array(d*[0])
        b = np.random.randn(d)
        b_history = [b]
        Q = 1*np.eye(d)
        etha = 1e5
        omega = 1
        count = 0

        while np.absolute(np.diag(Q).sum()) > self.delta:
            # Q violates properties of covariance matrix
            b, Q, rewards, thetas, etha, omega = self.__more_step__(b, Q, etha, omega)
            b_history += [b]
            count += 1
            print("Count: ", count, " Still improving...", np.diag(Q).sum())
            if (np.mod(count, 200) == 0) or (np.absolute(np.diag(Q).sum()) <= self.delta):
                plot(rewards, thetas, self.policy.get_number_of_parameters(), b_history)


    def __more_step__(self, b, Q, etha, omega):
        rewards, thetas = self.sample_generator.sample(b, Q)
        beta_hat = linear_regression(thetas, rewards)

        R, r, r0 = compute_quadratic_surrogate(beta_hat, np.asarray(thetas).shape[1])
        #print("R: ", R, " r: ", r, " r0: ", r0)
        for i, theta in enumerate(thetas):
            print(rewards[i] , " : ", theta @ R @ np.array([theta]).T + theta @ r + r0)
        # TODO: set diffrent epsilon, beta and start values for the optimization
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
        print("f: ", f)
        print("F: ", F)

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        print("parameter change: ", np.abs(b - b_new).sum())
        print("Reward max: ", max(rewards))
        print("Reward max - min: ", max(rewards) - min(rewards))
        print("theta = ", b_new)

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
