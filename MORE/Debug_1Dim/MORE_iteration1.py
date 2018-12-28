import numpy as np
from policy1 import *
from sample1 import *
from regression1 import * # , X
from optimization1 import *

class More(object):
    '''
        Does the MORE iterations until convergance is reached
        Convergance is archieved if the improvment from the prev iteration
        is smaller than delta
    '''

    def __init__(self, delta, policy, env):
        self.delta = delta
        self.sample_generator = Sample(env, policy)
        self.policy = policy


    def iterate(self):
        b, Q = 0, 1
        while np.absolute(Q) > self.delta:
            # Q violates properties of covariance matrix
            b, Q = self.__more_step__(b, Q)
            print("Still improving...", Q)


    def __more_step__(self, b, Q):
        # Generate samles for our policy
        # TODO: 10000,20,150 -> Übergeben
        rewards, thetas = self.sample_generator.sample(1, 20, 150, b, Q)

        beta_hat = linear_regression(thetas, rewards)

        # TODO: statt "np.asarray(thetas).shape[1]" - anders schreiben
        R, r, r0 = compute_quadratic_surrogate(beta_hat)
        # TODO: set diffrent epsilon, beta and start values for the optimization
        opti = Optimization(Q, b, R, r, .01, 0.99)
        etha0 = self.__compute_etha0__(1, Q, R)
        x0 = np.asarray([etha0, 1]) # starting point for etha and omega, where etha is large enough s.t. F is p.d.

        sol = opti.SLSQP(x0) #L_BFGS_B(x0)
        print("Computed etha: {}, omega: {}".format(sol.x[0], sol.x[1]))

        # Update pi
        etha = sol.x[0]
        omega = sol.x[1]
        F = 1/(etha / Q - 2 * R)
        f = (etha / Q) * b + r

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        print("parameter change: ", np.abs(b - b_new))
        print("Reward max - min: ", max(rewards) - min(rewards))
        print("theta = ", b_new)

        return b_new, Q_new

    def __compute_etha0__(self, etha0, Q, R):
        F = 1/(etha0/Q - 2 * R)

        while not F > 0:
            etha0 += 1
            F = 1/(etha0/Q - 2 * R)

        return etha0


        return etha0
