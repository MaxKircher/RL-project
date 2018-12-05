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

    def __init__(self, delta, policy, env):
        self.delta = delta
        self.sample_generator = Sample(env, policy)
        self.policy = policy


    def iterate(self):
        #### Raus in die test, cf. TODO
        d = self.policy.get_number_of_parameters()
        b = np.array(d*[0])
        Q = 10.*np.eye(d)
        ####

        # Abbruchbedingung -> Kaüitel 2 letzter Satz, asymptotic to point estimate
        while np.diag(Q).sum() > self.delta:
            b, Q = self.__more_step__(b, Q)
            print("Still improving...", np.diag(Q).sum())            


    def __more_step__(self, b, Q):
        # Generate samles for our policy
        # TODO: 10000,20,150 -> Übergeben
        rewards, thetas = self.sample_generator.sample(10000, 20, 150, b, Q)

        beta_hat = linear_regression(thetas, rewards)

        # TODO: statt "np.asarray(thetas).shape[1]" - anders schreiben
        R, r, r0 = compute_quadratic_surrogate(beta_hat, np.asarray(thetas).shape[1])
        # TODO: set diffrent epsilon, beta and start values for the optimization
        opti = Optimization(Q, b, R, r, 1, 0.99)
        x0 = np.ones(2) # starting point for etha and omega

        sol = opti.SLSQP(x0)
        print("Computed etha: {}, omega: {}".format(sol.x[0], sol.x[1]))

        # Update pi
        etha = 1#sol.x[0]
        omega = 1#sol.x[1]
        F = np.linalg.inv(etha * np.linalg.inv(Q) - 2 * R)
        f = etha * np.linalg.inv(Q) @ b + r

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        print("parameter change: ", np.abs(b - b_new).sum())
        print("Reward: ", max(rewards))

        return b_new, Q_new
