import numpy as np

class TrainingStates(object):
    
    def __init__(self, env):
        self.env = env

    def generate_training_states(self, number):
        s0 = self.env.reset()
        states = [s0]
        for j in range(number):
            s1, r, d, i = self.env.step(np.asarray(1))
            s2, r, d, i = self.env.step(np.asarray(-1))
            self.env.render()
            if d:
                s = self.env.reset()
                print("Resetted environment")
            states.append(s1)
            states.append(s2)

        return states
