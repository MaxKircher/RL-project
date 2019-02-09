import numpy as np

class GAE(object):

    def __init__(self, gamma, lambd):
        self.gamma = gamma
        self.lambd = lambd
        self.value = Value()

    def compute_td_residual(self, state, next_state, reward):
        return reward + self.gamma * self.value(next_state) - self.value(state)


    def compute_advantages(self, deltas):
        A = np.zeros(deltas.shape)
        A[-1] = self.gamma * self.lambd * deltas[-1]
        for i in range(deltas.shape - 2, -1, -1):
            A[i] = self.gamma * self.lambd * (deltas[i] + A[i+1])
        return A

    def update_value(self):
        self.value


def compute_discounted_rewards(rewards, gamma):
    R = np.zeros(rewards.shape)
    R[-1] = rewards[-1]
    for i in range(rewards.shape[0] - 2, -1, -1):
        R[i] = gamma * R[i + 1] + rewards[i]

    return R
