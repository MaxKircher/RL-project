import numpy as np
import gym
import quanser_robots
from trpo import *
from policy import *

env = gym.make('CartpoleStabShort-v0')
s0 = env.reset()
gamma = 0.99

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
#print(s_dim)
policy = NN(s_dim, a_dim)

trpo = TRPO(env, gamma, policy)


states, actions, Q = trpo.sample_sp(policy, s0, 1000)
g = trpo.compute_loss_gradients(states, actions, Q)

JM = trpo.compute_Jacobian(states)
FIM = trpo.compute_FIM_mean()

A = JM.T * FIM * JM # where A is the FIM w.r.t. to the Parameters theta see C
s = np.linalg.lstsq(A, g)[0]
beta = trpo.beta(0.01, np.matrix(s).T, A)
# print("Jacobi = ", JM)
# print("FIM = ", FIM)
print("A = ", A)
print("s = ", s)
print("beta = ", beta)



'''
    Line Search
    Parameterupdate unter Formel (17)
    Mehrdimensionale Action Spaces
'''
