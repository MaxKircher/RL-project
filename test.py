import numpy as np
import gym
import quanser_robots
from trpo import *
from policy import *

env = gym.make('Qube-v0')
s0 = env.reset()
gamma = 0.99

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
policy = NN(s_dim, a_dim)

trpo = TRPO(env, gamma, policy)


states, actions, Q = trpo.sample_sp(policy, s0, 100)
trpo.optimize(states, actions, Q)





#
# for i in range(states.shape[0]):
#     s = states[i]
#     a = actions[i]
#     Q[i]/policy.q(s, a)




#print(as)
#print(Q)
