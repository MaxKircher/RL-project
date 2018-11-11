import numpy as np
import gym
import quanser_robots
from policy import *

class TRPO(object):

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma



    def sample_sp(self, policy, s0, T): # sp = single path
        s = s0
        states = [s0] # leere Liste mit Actions
        actions = []
        rewards = []
        for i in range(T):
            a = policy.choose_a(s)
            s, r, done, info = self.env.step(a)
            states  += [s]
            actions += [a]
            rewards += [r]
            if done:
                T = i+1
                break

        states = np.array(states) # Aus Liste Array machen
        actions = np.array(actions)
        Q = np.zeros(T+1)
        for i in range(T-1, -1, -1):
            Q[i] = self.gamma*Q[i+1] + rewards[i]

        return states, actions, Q
