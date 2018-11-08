import numpy as np
import gym
import quanser_robots
from trpo import *
from policy import *

env = gym.make('Qube-v0')
s0 = env.reset()
gamma = 0.99
trpo = TRPO(env, gamma)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
policy = NN(s_dim, a_dim)


bla = trpo.sample_sp(policy, s0, 10)
print(bla)
#print(as)
#print(Q)
