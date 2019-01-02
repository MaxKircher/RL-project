import numpy as np

class TrainingStates(object):
    """
    Sample Training States to test if MORE converges for a static set of states
    """
    def __init__(self, env):
        self.env = env

    def generate_training_states(self, number):
        s0 = self.env.reset()
        states = [s0]
        for j in range(number):
            s, r, d, i = self.env.step(np.asarray(1))
            if d:
                s = self.env.reset()
                print("Resetted environment")
            states.append(s)
        return states
