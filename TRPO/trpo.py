import numpy as np
import gym
import quanser_robots
from policy import *
import torch

class TRPO(object):

    def __init__(self, env, gamma, policy):
        self.env = env
        self.gamma = gamma
        self.policy = policy



    def sample_sp(self, policy, s0, T): # sp = single path
        s = s0
        states = [s0] # leere Liste mit Actions
        actions = []
        rewards = []
        for i in range(T):
            a = policy.choose_a(s)
            s, r, done, info = self.env.step(a)
            # self.env.render()
            states  += [s]
            actions += [a]
            rewards += [r]
            # if done:
            #     T = i+1
            #     break

        print("Rewards: ", np.array(rewards).sum())

        states = np.array(states) # Aus Liste Array machen
        actions = np.array(actions)
        Q = np.zeros(T+1)
        for i in range(T-1, -1, -1):
            Q[i] = self.gamma*Q[i+1] + rewards[i]

        return states, actions, Q


    def arbitrary(self, pi, states, actions, Q):
        sum = torch.zeros(1).double()
        print("trpo.py/arbitrary states.shape[0] = ", states.shape[0])
        for i in range(states.shape[0]):
            s = states[i]
            a = actions[i]
            sum += pi(s, a)*Q[i]/torch.tensor(self.policy.q(s, a)).double()
            return -sum/states.shape[0] # - damit in optimize maximiert wird (ackward minimirt nämlich)


    def optimize(self, states, actions, Q):
        to_opt = self.arbitrary(self.policy.ableitbar_q, states, actions, Q)
        to_opt.backward()
