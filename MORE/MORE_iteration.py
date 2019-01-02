import numpy as np
from policy import *
from sample import *
from regression import * # , X
from optimization import *

class More(object):
    '''
        Does the MORE iterations until convergance is reached
        Convergance is archieved if the improvment from the prev iteration
        is smaller than delta
    '''

    def __init__(self, delta, policy, env, ts):
        self.delta = delta
        self.sample_generator = Sample(env, policy)
        self.policy = policy
        self.ts = ts


    def iterate(self):
        #### Raus in die test, cf. TODO
        d = self.policy.get_number_of_parameters()
        b = np.array(d*[0])
        Q = 1*np.eye(d)
        ####
        # Abbruchbedingung -> Kaüitel 2 letzter Satz, asymptotic to point estimate
        while np.absolute(np.diag(Q).sum()) > self.delta:
            # Q violates properties of covariance matrix
            b, Q = self.__more_step__(b, Q)
            print("Still improving...", np.diag(Q).sum())


    def __more_step__(self, b, Q):
        # Generate samles for our policy
        # TODO: 10000,20,150 -> Übergeben
        #rewards, thetas = self.sample_generator.sample(1000, 20, 150, b, Q)
        rewards, thetas = self.sample_generator.training_sample(20, b, Q, self.ts)

        beta_hat = linear_regression(thetas, rewards)

        # TODO: statt "np.asarray(thetas).shape[1]" - anders schreiben
        R, r, r0 = compute_quadratic_surrogate(beta_hat, np.asarray(thetas).shape[1])
        # TODO: set diffrent epsilon, beta and start values for the optimization
        opti = Optimization(Q, b, R, r, .01, 0.99)
        etha0 = self.__compute_etha0__(1, Q, R)
        x0 = np.asarray([etha0, 1]) # starting point for etha and omega, where etha is large enough s.t. F is p.d.

        sol = opti.L_BFGS_B(x0) #SLSQP(x0)
        print("Computed etha: {}, omega: {}".format(sol.x[0], sol.x[1]))

        # Update pi
        etha = sol.x[0]
        omega = sol.x[1]
        F = np.linalg.inv(etha * np.linalg.inv(Q) - 2 * R)
        f = etha * np.linalg.inv(Q) @ b + r

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        print("parameter change: ", np.abs(b - b_new).sum())
        print("Reward max: ", max(rewards))
        print("Reward max - min: ", max(rewards) - min(rewards))
        print("theta = ", b_new)

        return b_new, Q_new

    def __compute_etha0__(self, etha0, Q, R):
        F = np.linalg.inv(etha0 * np.linalg.inv(Q) - 2 * R)
        # print("inv(Q): ", np.linalg.inv(Q))
        while not np.all(np.linalg.eigvals(F) > 0):
            #print(etha0)
            etha0 += 1
            F = np.linalg.inv(etha0 * np.linalg.inv(Q) - 2 * R)

        # print("etha0 = ", np.linalg.eigvals(F) > 0)

        return etha0
