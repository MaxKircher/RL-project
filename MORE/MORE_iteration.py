import numpy as np
from policy import *
from sample import *
from regression import * # , X
from optimization import *

class MORE(object):
    '''
        Does the MORE iterations until convergance is reached
        Convergance is archieved if the improvment from the prev iteration
        is smaller than delta
    '''

    def __init__(self, delta, policy, env):
        self.delta = delta
        self.policy = policy
        self.env = env

    '''
        Dachte eine Abbruchbedingung für R reicht aus, oder wollen wir für R und Theta
        sicherstellen, dass die Verbesserung größer delta ist, bevor es zum Abbruch kommt?

        reward_old: integer
    '''
    def iterate(self, reward_old, Q, b):
        # Initalize sample generator
        sample_generator = SAMPLE(self.env, self.policy)

        # Generate samles for our policy
        rewards, thetas = sample_generator.sample(10, 3,  np.random.multivariate_normal, b, Q)

        # actually wo don't use a variable beta_hat_new, deswegen kann man die auch nur beta_hat bezeichnen(?)
        beta_hat_old = linear_regression(thetas, rewards)
        R, r, r0 = compute_quadratic_surrogate(beta_hat_old, np.asarray(thetas).shape[1])
        # TODO: set diffrent epsilon, beta and start values for the optimization
        opti = OPTIMIZATION(Q, b, R, r, 1, 1)
        x0 = np.ones(2) # starting point for etha and omega
        g = opti.objective(x0) # Entweder ca. 560 oder nan
        sol = opti.SLSQP(x0)

        # Update pi
        etha = 1 # sol.x[0]
        omega = 0 # sol.x[1]
        F = np.linalg.inv(etha * np.linalg.inv(Q) - 2 * R)
        f = etha * np.linalg.inv(Q) @ b + r

        b_new, Q_new = opti.update_pi(F, f, etha, omega)

        rewards_new, thetas_new = sample_generator.sample(10, 3, np.random.multivariate_normal, b_new, Q_new)
        print("update worked")

        # compute average reward over thetas
        reward_new = 0
        for rew in rewards_new:
            reward_new += rew
        # Does this make sense?
        reward_new = reward_new / len(rewards_new)

        # because reward_old is negative we exit right away. Set np.absolute
        # brakets diffrent
        if np.absolute((reward_new - reward_old) / reward_old) < self.delta:
            print("Found best thetas.") # wich of the 3 thetas do we return?
            return thetas_new[0] # maybe the one yielding the highest avg reward?
        else:
            print("Still improving...")
            return self.iterate(reward_new, Q_new, b_new)
