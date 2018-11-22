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
print("dim(A) = ", A.shape)
s = np.linalg.lstsq(A, g)[0]
beta = trpo.beta(0.01, np.matrix(s).T, A)

parameters = list(policy.model.parameters())
number_cols = sum(p.numel() for p in policy.model.parameters()) \
                + policy.model.log_dev.size(0) # size -> 0 oder 1
theta_old = torch.zeros(a_dim, number_cols)
j = 0

for param in parameters:
    theta_old[:,j: j + param.nelement()] = param.view(a_dim, -1)
    j += param.nelement()

theta_old[:,j] = policy.model.log_dev

# print("test: theta_old = ", theta_old)

policy_theta_new = trpo.line_search(beta, 0.1, s, theta_old, states, actions, Q)

print("policy_theta_new = ", policy_theta_new)

'''
    Line Search
    Parameterupdate unter Formel (17)
    Mehrdimensionale Action Spaces
'''
